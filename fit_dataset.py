import numpy as np
import pandas as pd
import pickle

from classes.agent import Agent
from classes.ou_network import OU_Network
from classes.internal_state import Normative_DIS, Local_computations_omniscient_DIS, Local_computations_interfocus_DIS
from classes.action_state import Discounted_gain_soft_horizon_TSAS, Undiscounted_gain_hard_horizon_TSAS
from classes.sensory_state import Omniscient_ST

from classes.experiment import Experiment

from methods.policies import softmax_policy_init
from methods.empirical_priors import discrete_empirical_priors
from methods.metrics import normalised_euclidean_distance


def main():
    ## Import behavioural experiment
    with open('/home/vbtesh/documents/CompProjects/vbtCogSci/csl2analysis/data/csl_2_modelling_data.obj', 'rb') as inFile:
        modelling_data = pickle.load(inFile)

    columns = ['generic', 'congruent', 'incongruent', 'implausible']
    datasets = {
        'normative': {
            'obs': pd.DataFrame(columns=columns),
            'random': pd.DataFrame(columns=columns)
        },
        'lc': {
            'obs': pd.DataFrame(columns=columns),
            'random': pd.DataFrame(columns=columns)
        }
    }


    for part_key in modelling_data.keys():

        part_data = modelling_data[part_key]['trials']
        

        for trial_name in part_data.keys():
        


            # Model fitting
            fitting = False # If false, no data will be used 

            ## Data from trial
            trial_data = part_data[trial_name]
            posterior = trial_data['posterior']
            data = trial_data['data']
            inters = trial_data['inters']
            inters_fit = trial_data['inters_fit']
            judgement_data = trial_data['links_hist']


            # General model parameters (true for all trials)
            if fitting:
                N = data.shape[0] - 1
                K = data.shape[1]
            else:
                N = 300
                K = 3
            links = np.array([-1, -0.5, 0, 0.5, 1])
            theta = 0.5
            dt = 0.2
            sigma = 3


            # Set up priors
            flat_prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            flat_prior = np.tile(flat_prior, (6, 1))

            random_prior = np.random.rand(6, 5)
            random_prior = random_prior / random_prior.sum(axis=1).reshape((6, 1))

            prior_perfect = np.array([[1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0]])
            # Emprical Priors
            if fitting and 'prior' in trial_data.keys():
                part_map = trial_data['prior'] # Participant's maximum a priori
                temp = 3/2 # Must be explored further
                empirical_priors, entropy = discrete_empirical_priors(part_map, links, temp)
            else:
                empirical_priors = random_prior

            ## Final prior assignment
            prior = empirical_priors

            # Ground truth model
            ## Import from behavioural experiment
            gt_behavioural_exp = trial_data['ground_truth']
            ## Any model as np.ndarray
            custom_model = np.array([-1, 0, .0, -1, 0, 0])
            ## Final ground truth assignment
            true_model = gt_behavioural_exp

            # Action parameters
            ## Number of model to sample from the posterior
            C = 3
            ## Different action possibility
            poss_actions = np.arange(-100, 100)
            poss_actions = np.array([85, 45, 0, -45, -85])
            poss_actions = np.arange(-100, 101, step=10)
            ## Define action length (TO BE REFINED FOR FITTING DATA)
            action_len = 5 
            ## Policy for action selection from action values
            policy_funcs = softmax_policy_init(1) # Returns three function: sample, pmf, params
            ## Parameters
            epsilon = 1e-2 # Certainty threshold: agent stops intervening after entropy goes below epsilon
            knowledge = False  # Can be a model as nd.array, True for perfect knowledge, 'random' for random sampling and False for posterior based sampling
            ## Special parameters for tree searches
            horizon = 1e-2 # For soft horizon discounted gain
            discount = 0.01 # For soft horizon discounted gain
            depth = 1 # Horizon for hard horizon undiscounted gain
            ## General behaviour parameter
            behaviour = 'obs'   # Can be 'obs', 'random' or 'actor'


            sensory_state = Omniscient_ST(N, K)
            external_state = OU_Network(N, K, true_model, theta, dt, sigma)

            for model in datasets.keys():
                if model == 'normative':
                    internal_state = Normative_DIS(N, K, prior, links, dt, theta, sigma, sample_params=False)
                    for act in datasets[model].keys():
                        behaviour = act
                        #action_state = Discounted_gain_soft_horizon_TSAS(N, K, behaviour, poss_actions, action_len, policy_funcs, epsilon, C, knowledge, discount, horizon)
                        action_state = Undiscounted_gain_hard_horizon_TSAS(N, K, behaviour, poss_actions, action_len, policy_funcs, epsilon, C, knowledge, depth)

                        agent = Agent(N, sensory_state, internal_state, action_state)

                        if fitting:
                            external_state.load_trial_data(data)
                            action_state.load_action_data(inters, inters_fit, data)
                            internal_state.load_judgement_data(judgement_data, posterior)

                        experiment = Experiment(agent, external_state)

                        # Run experiment
                        if fitting:
                            experiment.fit(posterior)
                        else:
                            experiment.run()

                        dist = normalised_euclidean_distance(true_model, internal_state.map)

                        datasets[model][act].loc[part_key, trial_name] = dist

                elif model == 'lc':
                    internal_state = Local_computations_omniscient_DIS(N, K, prior, links, dt, theta, sigma, sample_params=False)
                    #internal_state = Local_computations_interfocus_DIS(N, K, prior, links, dt, theta, sigma, sample_params=False)
                    for act in model.keys():
                        behaviour = act
                        #action_state = Discounted_gain_soft_horizon_TSAS(N, K, behaviour, poss_actions, action_len, policy_funcs, epsilon, C, knowledge, discount, horizon)
                        action_state = Undiscounted_gain_hard_horizon_TSAS(N, K, behaviour, poss_actions, action_len, policy_funcs, epsilon, C, knowledge, depth)

                        agent = Agent(N, sensory_state, internal_state, action_state)

                        if fitting:
                            external_state.load_trial_data(data)
                            action_state.load_action_data(inters, inters_fit, data)
                            internal_state.load_judgement_data(judgement_data, posterior)

                        experiment = Experiment(agent, external_state)

                        # Run experiment
                        if fitting:
                            experiment.fit(posterior)
                        else:
                            experiment.run()

                        dist = normalised_euclidean_distance(true_model, internal_state.map)

                        datasets[model][act].loc[part_key, trial_name] = dist


    for model in datasets.keys():
        for act in datasets[model].keys():
            datasets[model][act].to_csv(f'./data/{model}_{act}.csv')

if __name__ == '__main__':
    main()

pass

