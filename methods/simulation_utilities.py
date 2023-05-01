import numpy as np
import pandas as pd
import pickle
from os.path import exists
from scipy.optimize import minimize
import time
import math
import json
import gc

from classes.experiment import Experiment
from classes.agent import Agent
from methods.action_plans import fetch_action_plan, generate_action_plan
from methods.metrics import int_or_fl
from methods.states_params_importer import import_states_params_asdict

"""
General function for simulating data given specified parameters for each states

ARGS:
N (int): number of datapoints
K (int): number of variables

KWARGS:
external_state_params (dict): json style dict with the name of the state as keys and its parameters as a nested dict
internal_state_params (dict): json style dict with the name of the state as keys and its parameters as a nested dict
sensory_state_params (dict): json style dict with the name of the state as keys and its parameters as a nested dict
action_state_params (dict): json style dict with the name of the state as keys and its parameters as a nested dict
"""

def simulate(N, K, ground_truth, space_triple,
             external_state_params=None,
             internal_state_params=None,
             sensory_state_params=None,
             action_state_params=None,
             prior_judgement=None,
             realised_data=False,
             use_action_plan=None,
             trial_data=None,
             verbose=False):
    
    models_dict = import_states_params_asdict()
    

    """
    Establish used states
    """

    if not external_state_params:
        external_state_label = 'OU_Network'
    else:
        external_state_label = [e_s for e_s in external_state_params.keys()][0]
    
    if not internal_state_params: 
        internal_states_list = [
            #'normative_&_1',
            #'LC_discrete_&_1',
            #'LC_discrete_att_&_att',
            #'change_obs_fk_&_att_cha',
            'mean_field_vis'
        ]
    else:
        internal_states_list = [i_s for i_s in internal_state_params.keys()]

    if not sensory_state_params:
        sensory_states_list = [
            'omniscient'
        ]
    else:
        sensory_states_list = [s_s for s_s in sensory_state_params.keys()]
    
    if not action_state_params:
        action_states_list = [
            #'experience_vao',
            #'tree_search_soft_horizon',
            'tree_search_hard_horizon'
        ]
    else:
        action_states_list = [a_s for a_s in action_state_params.keys()]



    """
    Specify action plan if relevant
    """
    if use_action_plan:
        if isinstance(use_action_plan, float):
            action_plan = generate_action_plan(N, K, time=use_action_plan)
        elif use_action_plan == 'data':
            action_plan = [trial_data['inters'], trial_data['data']]
        elif isinstance(use_action_plan, str):
            action_plan = fetch_action_plan(use_action_plan, N, K)
        acting_len = (1 - np.isnan(action_plan[0])).mean()
    else:
        action_plan = None


    """
    Set up external state

    Amend default external state params if relevant 
    """
    external_state_kwargs = models_dict['external'][external_state_label]['params']['kwargs'].copy()
    if external_state_params and not realised_data:
        for param_key, param_val in external_state_params[external_state_label].items():
            if param_key in external_state_kwargs.keys():
                external_state_kwargs[param_key] = int_or_fl(param_val)
  
    external_state = models_dict['external'][external_state_label]['object'](N, K, 
                                                                     *models_dict['external'][external_state_label]['params']['args'],
                                                                     **external_state_kwargs, 
                                                                     ground_truth=ground_truth)
    
    if realised_data:
            external_state.load_trial_data(trial_data['data']) # Load Data
    
    
    """
    Set up internal state and sensory states
    """
    internal_states = []
    
    for i, model_tags in enumerate(internal_states_list):
        if len(model_tags.split('_&_')) == 2:
            model, tags = model_tags.split('_&_')
        else:
            model = model_tags.split('_&_')[0]

        ## Internal States
        internal_states_kwargs = models_dict['internal'][model]['params']['kwargs'].copy()
        # Setup fitted params
        if internal_state_params:
            for param_key, param_val in internal_state_params[model].items():
                # Internal states params
                if param_key in internal_states_kwargs.keys():
                    internal_states_kwargs[param_key] = int_or_fl(param_val)

        # Set up internal states
        pass
        i_s = models_dict['internal'][model]['object'](N, K, 
                                                       *models_dict['internal'][model]['params']['args'],
                                                       **internal_states_kwargs,
                                                       generate_sample_space = True)

        # Initialse space according to build_space
        i_s.add_sample_space_env(space_triple)

        # Initialise prior distributions for all IS
        i_s.initialise_prior_distribution(prior_judgement)
        
        internal_states.append(i_s)

    """
    Sensory states
    """
    sensory_states = []
    for i, model in enumerate(sensory_states_list):

        ## Sensory States
        sensory_states_kwargs = models_dict['sensory'][sensory_states_list[i]]['params']['kwargs'].copy()

        if sensory_state_params:
            for param_key, param_val in sensory_state_params[model].items():        
                # Sensory states params
                if param_key in sensory_states_kwargs.keys():
                    sensory_states_kwargs[param_key] = int_or_fl(param_val)

        # Set up sensory states
        sensory_s = models_dict['sensory'][sensory_states_list[i]]['object'](N, K, 
                                                                             *models_dict['sensory'][sensory_states_list[i]]['params']['args'],
                                                                             **sensory_states_kwargs)
        sensory_states.append(sensory_s)


    """
    Action states
    """
    action_states = []
    for model in action_states_list:
        action_states_kwargs = models_dict['actions'][model]['params']['kwargs'].copy()
        if action_state_params:
            for param_key, param_val in action_state_params[model].items():
                if param_key in action_states_kwargs.keys():
                    action_states_kwargs[param_key] = int_or_fl(param_val)

        a_s = models_dict['actions'][model]['object'](N, K, 
                                                     *models_dict['actions'][model]['params']['args'],
                                                     **action_states_kwargs)
        # Load action data if fitting
        if action_plan:
            a_s.load_action_plan(*action_plan)
        else:
            # If no action plan, behaviour is random
            a_s._behaviour = 'actor' # Can be random or actor

        action_states.append(a_s)

    if len(action_states) == 1: # Must be true atm, multiple action states are not supported
        action_states = action_states[0] 


    """
    Initialise agent
    """
    # Create agent
    if len(internal_states) == 1:
        agent = Agent(N, sensory_states[0], internal_states[0], action_states)
    else:
        agent = Agent(N, sensory_states, internal_states, action_states)

    
    """
    Initialise experiment
    """
    experiment = Experiment(agent, external_state)


    """
    Run experiment
    """
    experiment.run(verbose=verbose)

    """
    Delete high memory state
    """
    del action_states
    gc.collect()

    return experiment