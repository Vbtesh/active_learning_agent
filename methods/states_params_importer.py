import numpy as np

from classes.ou_network import OU_Network

from classes.internal_states.lc_omniscient_DIS import Local_computations_omniscient_DIS
from classes.internal_states.lc_omniscient_CIS import Local_computations_omniscient_CIS
from classes.internal_states.normative_DIS import Normative_DIS
from classes.internal_states.change_based_CIS import LC_linear_change_CIS
from classes.internal_states.change_based_DIS import LC_linear_change_DIS
from classes.internal_states.lc_interfocus_DIS import Local_computations_interfocus_DIS

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
            'change_continuous': LC_linear_change_CIS
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
prior_param = 1 # Priors param: temperature for discrete and variance form continuous /!\ Depends on the trial /!\
beta = None # Temperature for the softmax smothing function, will be fitted

## Attention based internal states
decay_rate = 0.65 # Discount rate of attention 
decay_type = 'sigmoid' # Functional form of attention parameter: 'exponential' or 'sigmoid'

## Normative and LC
### N/A: use OU parameters

## Change based internal states
prop_constant = theta*dt
samples_variance = 1e-1 # Variance of the likelihood of the links samples
hypothesis = 'full_knowledge' # can be 'distance', 'cause_value' and 'full_knowledge'


# SENSORY STATES parameters
change_memory = 2/3 # 1 means no smoothing, just look at raw change
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

# All but trial dependent stuff: Number of datapoint, Number of variables
def import_params_asdict():
    params_dict = {
        'internal': {
            'normative': {
                'args': [
                    L,
                    prior_param, # Prior (should be flat or depend on the empirical prior, needs more attention)
                    dt, 
                    theta,
                    sigma,
                ],
                'kwargs': {
                    'smoothing': beta
                }
            },
            'LC_discrete': {
                'args': [
                    L,
                    prior_param,
                    dt,
                    theta,
                    sigma
                ],
                'kwargs': {
                    'smoothing': beta
                }
            },
            'LC_continuous': {
                'args': [],
                'kwargs': {}
            },
            'LC_discrete_attention': {
                'args': [
                    L,
                    prior_param,
                    dt,
                    theta,
                    sigma,
                    decay_type,
                    decay_rate
                ],
                'kwargs': {
                    'smoothing': beta
                }
            },
            'change_discrete': {
                'args': [
                    L,
                    prior_param,
                    dt,
                    prop_constant,
                    samples_variance,
                    hypothesis,
                    decay_type,
                    decay_rate
                ],
                'kwargs': {
                    'smoothing': beta
                }
            },
            'change_continuous': {
                'args': [
                    L,
                    prior_param,
                    dt,
                    prop_constant,
                    samples_variance,
                    hypothesis,
                    decay_type,
                    decay_rate
                ],
                'kwargs': {
                    'smoothing': beta
                }
            },
        },
        'actions': {
            'tree_search_soft_horizon': {
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
            },
            'tree_search_hard_horizon': {
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
            },
            'experience_vao': {
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
            },
        },
        'sensory': {
            'omniscient': {
                'args': [],
                'kwargs': {
                    'alpha': change_memory,
                    'change': change_type
                }
            }
        }
    }
    return params_dict


def import_states_params_asdict():
    states_dict = import_states_asdict()
    params_dict = import_params_asdict()

    states_params_dict = {}
    for state_type, state_spec in states_dict.items():
        states_params_dict[state_type] = {}
        for state_name, state_data in state_spec.items():
            states_params_dict[state_type][state_name] = {}
            states_params_dict[state_type][state_name]['object'] = states_dict[state_type][state_name]
            states_params_dict[state_type][state_name]['params'] = params_dict[state_type][state_name]

    return states_params_dict
