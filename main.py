import numpy as np

from classes.agent import Agent
from classes.ou_network import OU_Network
from classes.internal_state import Normative_DIS
from classes.action_state import Discounted_gain_soft_horizon_TSAS
from classes.sensory_state import Omniscient_ST

from classes.experiment import Experiment

from methods.policies import softmax_policy_init

## Parameters
N = 100
K = 3
links = [-1, -0.5, 0, 0.5, 1]
theta = 0.5
dt = 0.2
sigma = 1 

flat_prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
flat_prior = np.tile(flat_prior, (6, 1))

## TRUE MODEL
true_model = np.array([-1, 0, 0, -1, 0, 0])

# Action parameters
C = 1

poss_actions = np.arange(-100, 100)
poss_actions = np.array([85, 45, 0, -45, -85])
#actions = np.arange(-100, 100)
#actions = np.array([-80, 80])
action_len = 5

policy, policy_pmf = softmax_policy_init(1)
#policy = epsilon_greedy_init(0.50)

idle = True
epsilon = 1e-2 # Certainty threshold: agent stops intervening after entropy goes below epsilon

knowledge = False  # Can be a model as nd.array, True for perfect knowledge, 'random' for random sampling and False for posterior based sampling
knowledge = true_model

horizon = 1e-1
discount = 0.1
depth = 2

behaviour = 'random'   # Can be 'obs', 'random' or 'actor'

## Initialise key objects
external_state = OU_Network(N, K, true_model, theta, dt, sigma)

sensory_state = Omniscient_ST(N, K)
action_state = Discounted_gain_soft_horizon_TSAS(N, K, behaviour, poss_actions, idle, action_len, policy, policy_pmf, epsilon, C, knowledge, discount, horizon)
internal_state = Normative_DIS(N, K, flat_prior, links, dt, theta, sigma, sample_params=True)

agent = Agent(N, sensory_state, internal_state, action_state)

experiment = Experiment(agent, external_state)

# Run experiment
experiment.run()

pass
