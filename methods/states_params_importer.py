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

from classes.action_states.discounted_gain_soft_horizon_TSAS import Discounted_gain_soft_horizon_TSAS
from classes.action_states.undiscounted_gain_hard_horizon_TSAS import Undiscounted_gain_hard_horizon_TSAS
from classes.action_states.experience_discrete_3D_AS import Experience_discrete_3D_AS
from classes.action_states.experience_conti_3D_AS import Experience_conti_3D_AS

from classes.sensory_states.omniscient_ST import Omniscient_ST
from methods.policies import softmax_policy_init
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
            'causal_event_segmentation': causal_event_segmentation_DIS
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
prior_param = 0 # Priors param: temperature for discrete and variance form continuous /!\ Depends on the trial /!\
beta = 40 # Temperature for the softmax smothing function, will be fitted

## Attention based internal states
decay_rate = 1 # Discount rate of attention 
decay_type = 'sigmoid' # Functional form of attention parameter: 'exponential' or 'sigmoid'

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

# SENSORY STATES parameters
change_memory = 1 # 1 means no smoothing, just look at raw change
change_type = 'raw' # Can be 'normalised', 'relative', 'raw'


# ACTION STATES parameters
behaviour = 'obs'   # Can be 'obs', 'random' or 'actor'
epsilon = 1e-2 # Certainty threshold: agent stops intervening after entropy goes below epsilon

## Tree search action states
tree_search_poss_actions = np.arange(-100, 101, step=10) # Possible actions to perform for tree search alg.
action_len = 1/dt # Length between each action selection in frames (1 second is baseline)
C = 3 # Number of model to sample from the posterior
knowledge = False  # Can be a model as nd.array, True for perfect knowledge, 'random' for random sampling and False for posterior based sampling
### Hard horizon
depth = 1 # Target depth for hard horizon undiscounted gain  
### Soft horizon
horizon = 1e-2 # For soft horizon discounted gain
discount = 0.01 # For soft horizon discounted gain
### Action selection policy
action_temperature = 1
softmax_policy_funcs = softmax_policy_init(action_temperature)
    
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
            (0, 500),
            ['prior_param']
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


    elif internal_state == 'LC_discrete_att':
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



    if fitting_prior:
        params_initial_guesses.append(params_dict['prior_param'][0])
        params_bounds.append(params_dict['prior_param'][1])
        internal_params_labels.append(params_dict['prior_param'][2] + [idx])
        fitting_list.append('prior')


    return (params_initial_guesses, params_bounds, internal_params_labels, action_params_labels, sensory_params_labels, fitting_list)

# All but trial dependent stuff: Number of datapoint, Number of variables
def import_states_params_asdict():
    params_dict = {
        'external': {
            'OU_Network': {
                'object': OU_Network,
                'params': {
                    'args': [
                        theta,
                        dt,
                        sigma
                    ],
                    'kwargs': {}
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
                        sigma,
                    ],
                    'kwargs': {
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
                        discount,
                        horizon
                    ],
                    'kwargs': {}
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
                        depth
                    ],
                    'kwargs': {}
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
                        'change_memory': change_memory,
                        'change': change_type
                    }
                }     
            }
        }
    }
    return params_dict

