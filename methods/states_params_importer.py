from statistics import variance
import numpy as np
from classes.action_states.action_state import Treesearch_AS

from classes.ou_network import OU_Network

from classes.internal_states.lc_omniscient_DIS import Local_computations_omniscient_DIS
from classes.internal_states.lc_omniscient_CIS import Local_computations_omniscient_CIS
from classes.internal_states.normative_DIS import Normative_DIS
from classes.internal_states.change_based_CIS import LC_linear_change_CIS
from classes.internal_states.change_based_DIS import LC_linear_change_DIS
from classes.internal_states.lc_interfocus_DIS import Local_computations_interfocus_DIS
from classes.internal_states.causal_event_segmentation_DIS import causal_event_segmentation_DIS
from classes.internal_states.mean_field_VIS import MeanField_VIS

from classes.action_states.discounted_gain_soft_horizon_TSAS import Discounted_gain_soft_horizon_TSAS
from classes.action_states.undiscounted_gain_hard_horizon_TSAS import Undiscounted_gain_hard_horizon_TSAS
from classes.action_states.experience_discrete_3D_AS import Experience_discrete_3D_AS
from classes.action_states.experience_conti_3D_AS import Experience_conti_3D_AS

from classes.sensory_states.omniscient_ST import Omniscient_ST
from methods.policies import epsilon_greedy_init, softmax_policy_init
from methods.policies import discrete_policy_init


def import_states_asdict():
    states_dict = {
        'internal': {
            'normative': Normative_DIS,
            'LC_discrete': Local_computations_omniscient_DIS,
            'LC_continuous': Local_computations_omniscient_CIS,
            'LC_discrete_attention': Local_computations_interfocus_DIS,
            'change_discrete': LC_linear_change_DIS,
            'change_continuous': LC_linear_change_CIS,
            'causal_event_segmentation': causal_event_segmentation_DIS,
            'mean_field_vis': MeanField_VIS
        },
        'actions': {
            'tree_search_soft_horizon': Discounted_gain_soft_horizon_TSAS,
            'tree_search_hard_horizon': Undiscounted_gain_hard_horizon_TSAS,
            'experience_vao': Experience_conti_3D_AS
        },
        'sensory': {
            'omniscient': Omniscient_ST
        }
    }

    return states_dict


# All but trial dependent stuff: Number of datapoint, Number of variables
def import_states_params_asdict():
    # GENERAL parameters
    N = 301 # Number of datapoints
    K = 3 # Number of variables in the network

    # OU parameters
    theta = 0.5 # rigidity
    sigma = 3 # Variance of the one step white noise
    dt = 1/5 # Time step: 1/dt = fps


    # EXTERNAL STATE parameters
    ground_truth = np.zeros(K**2 - K)  # Ground truth model generating the data: /!\ Depends on the trial /!\


    # INTERNAL STATE parameters
    L = np.array([-1, -1/2, 0, 1/2, 1]) # Possible link values
    #L = np.array([-1, 0, 1])
    prior_param = 0 # Priors param: temperature for discrete and variance form continuous /!\ Depends on the trial /!\
    beta = 1 # Temperature for the softmax smoothing function, will be fitted

    ## Attention based internal states
    decay_rate = 0.0 # Discount rate of attention 
    decay_type = 'exponential' # Functional form of attention parameter: 'exponential' or 'sigmoid'

    ## Normative and LC
    ### N/A: use OU parameters

    ## Change based internal states
    prop_constant = theta*dt
    samples_variance = 3/5 # Variance of the likelihood of the links samples
    hypothesis = 'full_knowledge' # can be 'distance', 'cause_value' and 'full_knowledge'

    # Causal event segmentation
    abs_bounds = (0, 100)
    ces_type = 'time_sensitive'
    ce_threshold = 0.3 # Causal event threshold (read as a percentage of bounds)
    time_threshold = 10  # The time threshold in frames before is strong, after is weak
    guess = 0.3 # The probability mass to be shared among other possibilities
    beta_ces = 4

    # Variational agent
    # Type agent:
    ## normative VS local_computations
    factorisation = 'normative' 
    ## Different update schedules:
    ### omniscient, full, single_factor, single_link, single_variable
    update_schedule = 'full'
    certainty_threshold = .4 # Should be represented as a percentage of the maximum entropy
    evidence_weight = 1.0
    block_learning = [
        #'theta'
    ]
    theta_values = np.array([0.1, 0.5, 1, 2])
    #theta_values = np.array([0.1, 0.5, 1])
    sigma_values = np.array([sigma/2, sigma, sigma*2, sigma*4], dtype=float)
    #sigma_values = np.array([1, 2, 3, 4])
    theta_prior = np.ones(theta_values.size) / theta_values.size
    sigma_prior = np.ones(4) / 4
    #theta_prior = np.array([0.9, 0.05, 0.05])
    #sigma_prior = np.array([0.0033, 0.99, 0.0033, 0.0033])
    parameter_set_init = {
        'theta': {
            'values': theta_values,
            'prior': theta_prior,
            'type': 'no_link'
        },
        'sigma': {
            'values': sigma_values,
            'prior': sigma_prior,
            'type': 'no_link'
        },
    }

    # SENSORY STATES parameters
    noise_std = 0 * dt
    change_memory = 1 # 1 means no smoothing, just look at raw change
    change_type = 'raw' # Can be 'normalised', 'relative', 'raw'


    # ACTION STATES parameters
    behaviour = 'obs'   # Can be 'obs', 'random' or 'actor'
    epsilon = 0.01 # Certainty threshold: agent stops intervening after entropy goes below epsilon

    ## Tree search action states
    #np.arange(-100, 101, step=50)
    tree_search_poss_actions = np.arange(-100, 101, step=25) # Possible actions to perform for tree search alg.
    tree_search_poss_actions = np.arange(-100, 101, step=50)
    action_len = 1#1/dt # Length between each action selection in frames (1 second is baseline)
    C = 5 # Number of model to sample from the posterior
    knowledge = 'posterior_unweighted'  # Can be a model as nd.array, 'posterior_unweighted', 'posterior_weighted', 'perfect' for perfect knowledge and 'random' for random sampling
    # Gain parameters
    gain_type = 'expected_information_gained'
    resource_rational_parameter = 0.1
    ### Hard horizon
    depth = 1 # Target depth for hard horizon undiscounted gain  
    ### Soft horizon
    horizon = 1e-2 # For soft horizon discounted gain
    discount = 0.01 # For soft horizon discounted gain
    ### Action selection policy
    action_temperature = 1
    softmax_policy_funcs = epsilon_greedy_init(0)#softmax_policy_init(action_temperature)



    ## Experience Value Acting Observing (vao) action states
    experience_poss_actions = np.arange(0, 101, step=10) # Possible actions to perform for experiences
    mus = np.array([80, 6, 2]) # Mus are the means: [value, acting_len, obs_len], note: acting and obs len should be in seconds
    cov = np.array([[20**2, 0, 0],
                   [0, 1**2, 0],
                   [0, 0, 1**2]])
    actions_prior_params = (mus, cov) # Continuous Gaussian parameters for the distribution over interventions
    time_unit = dt
    action_learn_rate = 0 # 0 means no learning, 1 means do last sampled action (random walk)
    max_acting_time = 30 # Maximum acting time in seconds
    max_obs_time = 30 # Maximum observing time in seconds
    experience_measure = 'information' # Can be "information" or "change"
    ### Action selection policy
    discrete_policy_funcs = discrete_policy_init()


    params_dict = {
        'external': {
            'OU_Network': {
                'object': OU_Network,
                'params': {
                    'args': [
                        dt
                    ],
                    'kwargs': {
                        'theta': theta,
                        'sigma': sigma,
                    }
                }
            }
        },
        'internal': {
            'normative': {
                'object': Normative_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        theta,
                        sigma
                    ],
                    'kwargs': {
                        'evidence_weight': evidence_weight,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
                
            },
            'LC_discrete': {
                'object': Local_computations_omniscient_DIS,
                'params': {
                    'args': [
                        L, 
                        dt, 
                        theta,
                        sigma,
                    ],
                    'kwargs': {
                        'evidence_weight': evidence_weight,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
            },
            'LC_continuous': {
                'object': Local_computations_omniscient_CIS,
                'params': {
                    'args': [],
                    'kwargs': {}
                }     
            },
            'LC_discrete_attention': {
                'object': Local_computations_interfocus_DIS,
                'params': {
                    'args': [
                        L, 
                        dt, 
                        theta,
                        sigma,
                        decay_type  
                    ],
                    'kwargs': {
                        'decay_rate': decay_rate,
                        'evidence_weight': evidence_weight,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
            },
            'LC_discrete_att': {
                'object': Local_computations_interfocus_DIS,
                'params': {
                    'args': [
                        L, 
                        dt, 
                        theta,
                        sigma,
                        'exponential'  
                    ],
                    'kwargs': {
                        'decay_rate': decay_rate,
                        'evidence_weight': evidence_weight,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
            },
            'LC_discrete_att_all': {
                'object': Local_computations_interfocus_DIS,
                'params': {
                    'args': [
                        L, 
                        dt, 
                        theta,
                        sigma,
                        'exponential'  
                    ],
                    'kwargs': {
                        'decay_rate': decay_rate,
                        'evidence_weight': evidence_weight,
                        'prior_param': prior_param,
                        'smoothing': beta,
                        'varfocus': False
                    }
                }
            },
            'Adaptive_LC': {
                'object': Local_computations_interfocus_DIS,
                'params': {
                    'args': [
                        L, 
                        dt, 
                        theta,
                        sigma,
                        'exponential'  
                    ],
                    'kwargs': {
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta,
                        'varfocus': False
                    }
                }
            },
            'Adaptive_Selective_LC': {
                'object': Local_computations_interfocus_DIS,
                'params': {
                    'args': [
                        L, 
                        dt, 
                        theta,
                        sigma,
                        'exponential'  
                    ],
                    'kwargs': {
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
            },
            'change_d_obs_fk': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'full_knowledge',
                        decay_type  
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }   
            },
            'change_d_obs_cause_effect': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'cause_effect_values',
                        decay_type 
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }   
            },
            'change_d_obs_cause': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'cause_value',
                        decay_type  
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
                
            },
            'change_d_obs_dist': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'distance',
                        decay_type  
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }    
            },
            'change_obs_fk': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'full_knowledge',
                        'exponential'  
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }   
            },
            'change_obs_cause_effect': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'cause_effect_values',
                        'exponential' 
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }   
            },
            'change_obs_cause': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'cause_value',
                        'exponential'  
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
                
            },
            'change_obs_dist': {
                'object': LC_linear_change_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        'distance',
                        'exponential'  
                    ],
                    'kwargs': {
                        'lh_var': samples_variance,
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }    
            },
            'ces_strength': {
                'object': causal_event_segmentation_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        abs_bounds,
                        'strength_sensitive'
                    ],
                    'kwargs': {
                        'ce_threshold': ce_threshold,
                        'time_threshold': time_threshold,
                        'prior_param': prior_param,
                        'guess': guess
                    }
                }

            },
            'ces_strength_unrestricted': {
                'object': causal_event_segmentation_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        abs_bounds,
                        'strength_sensitive'
                    ],
                    'kwargs': {
                        'ce_threshold': ce_threshold,
                        'time_threshold': time_threshold,
                        'prior_param': prior_param,
                        'guess': guess
                    }
                }

            },
            'ces_strength_softmax': {
                'object': causal_event_segmentation_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        abs_bounds,
                        'strength_sensitive'
                    ],
                    'kwargs': {
                        'ce_threshold': ce_threshold,
                        'prior_param': prior_param,
                        'smoothing': beta_ces
                    }
                }
            },
            'ces_no_strength': {
                'object': causal_event_segmentation_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        abs_bounds,
                        'strength_insensitive'
                    ],
                    'kwargs': {
                        'ce_threshold': ce_threshold,
                        'guess': guess,
                        'prior_param': prior_param
                    }
                }
            },
            'ces_no_strength_softmax': {
                'object': causal_event_segmentation_DIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        abs_bounds,
                        'strength_insensitive'
                    ],
                    'kwargs': {
                        'ce_threshold': ce_threshold,
                        'prior_param': prior_param,
                        'smoothing': beta_ces
                    }
                }
            },
            'change_continuous': {
                'object': LC_linear_change_CIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        prop_constant,
                        samples_variance,
                        hypothesis,
                        decay_type,
                        decay_rate  
                    ],
                    'kwargs': {
                        'decay_rate': decay_rate,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
                
            },
            'mean_field_vis': {
                'object': MeanField_VIS,
                'params': {
                    'args': [
                        L,
                        dt,
                        parameter_set_init
                    ],
                    'kwargs': {
                        'factorisation': factorisation,
                        'update_schedule': update_schedule,
                        'evidence_weight': evidence_weight,
                        'certainty_threshold': certainty_threshold,
                        'block_learning': block_learning,
                        'prior_param': prior_param,
                        'smoothing': beta
                    }
                }
                
            },
        },
        'actions': {
            'tree_search_soft_horizon': {
                'object': Discounted_gain_soft_horizon_TSAS,
                'params': {
                    'args': [
                        behaviour,
                        epsilon,
                        tree_search_poss_actions,
                        action_len,
                        softmax_policy_funcs,
                        C,
                        knowledge,
                        gain_type,
                        discount,
                        horizon
                    ],
                    'kwargs': {
                        'resource_rational_parameter': resource_rational_parameter
                    }
                }
            },
            'tree_search_hard_horizon': {
                'object': Undiscounted_gain_hard_horizon_TSAS,
                'params': {
                    'args': [
                        behaviour,
                        epsilon,
                        tree_search_poss_actions,
                        action_len,
                        softmax_policy_funcs,
                        C,
                        knowledge,
                        gain_type,
                        depth
                    ],
                    'kwargs': {
                        'resource_rational_parameter': resource_rational_parameter
                    }
                }
                
            },
            'experience_vao': {
                'object': Experience_discrete_3D_AS,
                'params': {
                    'args': [
                        behaviour,
                        epsilon,
                        experience_poss_actions,
                        discrete_policy_funcs, # Not right
                        time_unit,
                        max_acting_time,
                        max_obs_time,
                        experience_measure,
                        actions_prior_params,
                        action_learn_rate
                    ],
                    'kwargs': {}
                }
            }
                
        },
        'sensory': {
            'omniscient': {
                'object': Omniscient_ST,
                'params': {
                    'args': [],
                    'kwargs': {
                        'noise_std': noise_std,
                        'change_memory': change_memory,
                        'change': change_type
                    }
                }     
            }
        }
    }
    return params_dict




#
def params_to_fit_importer(internal_state, 
                           fitting_change=True, 
                           fitting_attention=True,
                           fitting_guess=True,
                           fitting_strength=True,
                           fitting_prior=True,
                           random_increment=1):


    params_dict = {
        'smoothing': [
            1,
            (0, 100),
            ['smoothing']
        ],
        'decay_rate': [
            2/3 + 1e-1*np.random.normal() * random_increment,
            (0, 1),
            ['decay_rate']
        ],
        'change_memory': [
            1/2 + 1e-1*np.random.normal() * random_increment,
            (1/10, 1),
            ['change_memory']
        ],
        'ce_threshold' : [
            1/5 + 1e-1*np.random.normal() * random_increment,
            (1e-2, 1),
            ['ce_threshold']
        ],
        'time_threshold' : [
            15 + 2*np.random.normal() * random_increment, 
            (1, 50),
            ['time_threshold']
        ],
        'guess': [
            0.1 + 1e-1*np.random.normal() * random_increment,
            (1e-2, 0.7),
            ['guess']
        ],
        'prior_param': [
            1 + 1e-1*np.random.normal() * random_increment,
            (0, 550),
            ['prior_param']
        ],
        'certainty_threshold': [
            np.abs(0.1 + 1e-2*np.random.normal() * random_increment),
            (1e-2, 10),
            ['certainty_threshold']
        ]
    }

    params_initial_guesses = [] 
    params_bounds = []
    internal_params_labels = []
    action_params_labels = []
    sensory_params_labels = []
    fitting_list = []
    idx = 0

    if internal_state[:6] == 'change':
        
        params_initial_guesses.append(params_dict['smoothing'][0])
        params_bounds.append(params_dict['smoothing'][1])
        internal_params_labels.append(params_dict['smoothing'][2] + [idx])
        idx += 1
        
        if fitting_attention:
            params_initial_guesses.append(params_dict['decay_rate'][0])
            params_bounds.append(params_dict['decay_rate'][1])
            internal_params_labels.append(params_dict['decay_rate'][2] + [idx])
            fitting_list.append('att')
            idx += 1
        
        if fitting_change:
            params_initial_guesses.append(params_dict['change_memory'][0])
            params_bounds.append(params_dict['change_memory'][1])
            sensory_params_labels.append(params_dict['change_memory'][2] + [idx])
            fitting_list.append('cha')
            idx += 1

    elif internal_state[:3] == 'ces':
        params_initial_guesses.append(params_dict['ce_threshold'][0])
        params_bounds.append(params_dict['ce_threshold'][1])
        internal_params_labels.append(params_dict['ce_threshold'][2] + [idx])
        idx += 1

        if fitting_strength:
            params_initial_guesses.append(params_dict['time_threshold'][0])
            params_bounds.append(params_dict['time_threshold'][1])
            internal_params_labels.append(params_dict['time_threshold'][2] + [idx])
            fitting_list.append('str')
            idx += 1

        if fitting_guess:
            params_initial_guesses.append(params_dict['guess'][0])
            params_bounds.append(params_dict['guess'][1])
            internal_params_labels.append(params_dict['guess'][2] + [idx])
            fitting_list.append('guess')
            idx += 1


    elif internal_state == 'LC_discrete':
        params_initial_guesses.append(params_dict['smoothing'][0])
        params_bounds.append(params_dict['smoothing'][1])
        internal_params_labels.append(params_dict['smoothing'][2] + [idx])
        idx += 1


    elif 'LC_discrete_att' in internal_state or 'Adaptive' in internal_state:
        params_initial_guesses.append(params_dict['smoothing'][0])
        params_bounds.append(params_dict['smoothing'][1])
        internal_params_labels.append(params_dict['smoothing'][2] + [idx])
        idx += 1
        
        if fitting_attention:
            params_initial_guesses.append(params_dict['decay_rate'][0])
            params_bounds.append(params_dict['decay_rate'][1])
            internal_params_labels.append(params_dict['decay_rate'][2] + [idx])
            fitting_list.append('att')
            idx += 1

    elif internal_state == 'normative':
        params_initial_guesses.append(params_dict['smoothing'][0])
        params_bounds.append(params_dict['smoothing'][1])
        internal_params_labels.append(params_dict['smoothing'][2] + [idx])
        idx += 1

    elif internal_state == 'mean_field_vis':
        
        params_initial_guesses.append(params_dict['smoothing'][0])
        params_bounds.append(params_dict['smoothing'][1])
        internal_params_labels.append(params_dict['smoothing'][2] + [idx])
        idx += 1

        params_initial_guesses.append(params_dict['certainty_threshold'][0])
        params_bounds.append(params_dict['certainty_threshold'][1])
        internal_params_labels.append(params_dict['certainty_threshold'][2] + [idx])
        idx += 1


    if fitting_prior:
        params_initial_guesses.append(params_dict['prior_param'][0])
        params_bounds.append(params_dict['prior_param'][1])
        internal_params_labels.append(params_dict['prior_param'][2] + [idx])
        fitting_list.append('prior')


    return (params_initial_guesses, params_bounds, internal_params_labels, action_params_labels, sensory_params_labels, fitting_list)
