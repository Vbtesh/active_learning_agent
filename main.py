import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json

from classes.experiment import Experiment
from classes.agent import Agent

from methods.states_params_importer import import_states_params_asdict
from methods.action_plans import generate_action_plan

from classes.action_states.discounted_gain_soft_horizon_TSAS import Discounted_gain_soft_horizon_TSAS
from classes.action_states.undiscounted_gain_hard_horizon_TSAS import Undiscounted_gain_hard_horizon_TSAS


## Import behavioural experiment
with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
    modelling_data = pickle.load(inFile)

print(len(modelling_data.keys()))

models_dict = import_states_params_asdict()
# Change to default parameters

# active states
#models_dict['actions']['tree_search_hard_horizon']['params']['args'][-2] = 'resource_rational'#'expected_information_gained'
#models_dict['actions']['tree_search_hard_horizon']['params']['kwargs']['resource_rational_parameter'] = 1

part_key = '5fb91837b8c8756d924f7351'
conditions = ['generic_0', 'congruent', 'incongruent', 'implausible']
cond = conditions[0]

# Model fitting
fit_or_run = False # If false, no data will be used 


part_data = modelling_data[part_key]
part_experiment = part_data['experiment']
trials = part_data['trials']

trial_data = trials[cond]
## Data from trial
model_name = trial_data['name'][:-2]
difficulty = cond
data = trial_data['data'] # Raw numerical data of variable values
ground_truth = trial_data['ground_truth'] # Ground truth model from which data has been generated
ground_truth = np.array([1, 0, 0, 1, 0, 0])
inters = trial_data['inters'] # Interventions as is
inters_fit = trial_data['inters_fit'] # Interventions with removed movements
judgement_data = trial_data['links_hist'] # Change in judgement sliders
posterior_judgement = trial_data['posterior'] # Final states of judgement sliders
prior_judgement = trial_data['prior'] if 'prior' in trial_data.keys() else None
utid = trial_data['utid']



use_fitted_parameters = True
if use_fitted_parameters:
    with open('./data/params_fitting_outputs/fitted_params.json', 'r') as infile:
        use_fitted_params = json.load(infile)

if use_fitted_params:
    fitted_params_dict = use_fitted_params[part_key] 

internal_states_list = [
    #'normative_&_1',
    #'LC_discrete_&_1',
    #'LC_discrete_att_&_att',
    #'change_obs_fk_&_att_cha',
    'mean_field_vis'
]
sensory_states_list = [
    'omniscient'
]
action_states_list = [
    #'experience_vao',
    #'tree_search_soft_horizon',
    'tree_search_hard_horizon'
]

use_action_plan = None

# Unpack generic trial relevant parameters
N = data.shape[0] # Number of datapoints
K = data.shape[1] # Number of variable

if use_action_plan:
    if isinstance(use_action_plan, float):
        action_plan = generate_action_plan(N, K, time=use_action_plan)
    elif isinstance(use_action_plan, str):
        action_plan = [trial_data['inters'], trial_data['data']]
    acting_len = (1 - np.isnan(action_plan[0])).mean()
else:
    action_plan = None

# Set up OU netowrk 
external_state = models_dict['external']['OU_Network']['object'](N, K, 
                                                                 *models_dict['external']['OU_Network']['params']['args'],
                                                                 **models_dict['external']['OU_Network']['params']['kwargs'], 
                                                                 ground_truth=ground_truth)

if fit_or_run == 'run':
    external_state.load_trial_data(data) # Load Data

# Set up states
## Internal states and sensory states
internal_states = []
sensory_states = []
for i, model_tags in enumerate(internal_states_list):
    if len(model_tags.split('_&_')) == 2:
        model, tags = model_tags.split('_&_')
    else:
        model = model_tags.split('_&_')[0]

    ## Internal States
    internal_states_kwargs = models_dict['internal'][model]['params']['kwargs'].copy()
    ## Sensory States
    sensory_states_kwargs = models_dict['sensory'][sensory_states_list[i]]['params']['kwargs'].copy()
    # Setup fitted params
    if use_fitted_params and model_tags in fitted_params_dict.keys():
        for param_key, param_val in fitted_params_dict[model_tags].items():
            # Internal states params
            if param_key in internal_states_kwargs.keys():
                internal_states_kwargs[param_key] = param_val
            # Sensory states params
            if param_key in sensory_states_kwargs.keys():
                sensory_states_kwargs[param_key] = param_val

    # Set up internal states
    i_s = models_dict['internal'][model]['object'](N, K, 
                                                   *models_dict['internal'][model]['params']['args'],
                                                   **internal_states_kwargs,
                                                   generate_sample_space = True)

    # Initialise prior distributions for all IS
    i_s.initialise_prior_distribution(prior_judgement)
    # Load data if fitting
    if fit_or_run == 'fit':
        i_s.load_judgement_data(judgement_data, posterior_judgement, False)
    
    internal_states.append(i_s)
        
    # Set up sensory states
    sensory_s = models_dict['sensory'][sensory_states_list[i]]['object'](N, K, 
                                                                         *models_dict['sensory'][sensory_states_list[i]]['params']['args'],
                                                                         **sensory_states_kwargs)
    sensory_states.append(sensory_s)

## Action states
action_states = []
for model in action_states_list:
    action_states_kwargs = models_dict['actions'][model]['params']['kwargs'].copy()
    if use_fitted_params and model in fitted_params_dict.keys():
        for param_key, param_val in fitted_params_dict[model].items():
            if param_key in action_states_kwargs.keys():
                action_states_kwargs[param_key] = param_val
    
    a_s = models_dict['actions'][model]['object'](N, K, 
                                                 *models_dict['actions'][model]['params']['args'],
                                                 **action_states_kwargs)
    # Load action data if fitting
    if fit_or_run == 'fit':
        a_s.load_action_data(inters, inters_fit, data)
    else:
        if action_plan:
            a_s.load_action_plan(*action_plan)
        else:
            # If no action plan, behaviour is random
            a_s._behaviour = 'actor' # Can be random or actor

    action_states.append(a_s)

if len(action_states) == 1: # Must be true atm, multiple action states are not supported
    action_states = action_states[0] 

        
# Create agent
if len(internal_states) == 1:
    agent = Agent(N, sensory_states[0], internal_states[0], action_states)
else:
    agent = Agent(N, sensory_states, internal_states, action_states)

# Create experiment
experiment = Experiment(agent, external_state)


# Run experiment
verbose = True
if fit_or_run == 'fit':
    experiment.fit(verbose=verbose)
else:
    experiment.run(verbose=verbose)

pass
#experiment.entropy_report()