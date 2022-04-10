from sys import intern
import numpy as np
import pandas as pd
import pickle

from classes.experiment import Experiment
from classes.agent import Agent

from methods.sample_space_methods import build_space_env

# Fit models to data, independent of participant's judgement
## No smoothing nor temperature parameter

# Returns: None
# Saves to file
## df: a DataFrame of summary statistics
## posteriors: a Dataframe of posteriors for all trials
## 1 file per model
### full distribution over models
##### => Can be represented by df, easy to compute marginals: cols = [link1, link2, ..., link6, p(m)_utid_1, p(m)_utid_2, ..., p(m)_utid_1200]

## posterior history: a collection of matrices representing posterior distributions throughout the trial
### full distribution over models 
##### => Can be represented by df, easy to compute marginals: cols = [link1, link2, ..., link6, p(m)_0, p(m)_1, ..., p(m)_N]
### marginal over links

def fit_models(internal_states_list,                # List of internal states names as strings
               action_states_list,                  # List of action states names as strings
               sensory_states_list,                 # List of sensory states names as strings
               External_state,                      # External state name: OU_netowrk
               models_dict,                         # Dict of model object and parameters: can be changed in separate file                 
               data_dict,                           # Dict of data for each participants/trial/experiment
               build_space=True,                    # Boolean, set true for pre initialisation of fixed and large objects
               fit_judgement=False):                # List of internal states names as strings

    # General loop
    ## Initialise general invariant parameters
    K = 3
    links = np.array([-1, -0.5, 0, 0.5, 1]) # Possible link values
    theta = 0.5 # Rigidity parameters
    dt = 0.2 # Time step
    sigma = 3 # Standard deviation

    # Build internal sample space
    if build_space:
        space_triple = build_space_env(K, links)

    
    # Define DataFrame
    cols = ['utid', 'pid', 'experiment', 'difficulty', 'scenario', 'model_name', 'ground_truth', 'posterior_map', 'posterior_judgement', 'prior_judgement', 'prior_entropy', 'posterior_entropy', 'model_specs']
    df = pd.DataFrame(columns=cols)

    # Posterior DataFrame
    # One per internal state
    cols = [f'link_{i}' for i in range(K)]
    dfs_posteriors = [pd.DataFrame(columns=links, data=space_triple[0]) for model in internal_states_list]


    # Count participant index
    sample_size = len(data_dict.keys())
    part_idx = 0

    for participant, part_data in data_dict.items():
        # Participant metadata
        part_experiment = part_data['experiment']
        trials = part_data['trials']

        for trial_type, trial_data in trials.items():
            
            # Extract data from participant's trial
            model_name = trial_data['name'][:-2]
            difficulty = trial_type
            data = trial_data['data'] # Raw numerical data of variable values
            ground_truth = trial_data['ground_truth'] # Ground truth model from which data has been generated
            inters = trial_data['inters'] # Interventions as is
            inters_fit = trial_data['inters_fit'] # Interventions with removed movements
            judgement_data = trial_data['link_hist'] # Change in judgement sliders
            posterior_judgement = trial_data['posterior'] # Final states of judgement sliders
            prior_judgement = trial_data['prior'] if 'prior' in trial_data.keys() else None

            # Unpack generic trial relevant parameters
            N = data.shape[0] # Number of datapoints
            K = data.shape[1] # Number of variables

            # Set up OU netowrk 
            external_state = External_state(N, K, theta, dt, sigma, ground_truth=ground_truth)
            external_state.load_trial_data(data) # Load Data

            # Set up states
            ## Internal states
            internal_states = []   
            for model in internal_states_list:
                i_s = models_dict['internal'][model]['object'](N, K, 
                                                               *models_dict['internal'][model]['params']['args'],
                                                               **models_dict['internal'][model]['params']['kwargs'],
                                                               generate_space_space = not build_space)

                # Initialse space according to build_space
                if build_space:
                    i_s.add_sample_space_env(space_triple)

                # Initialise prior distributions for all IS
                i_s.initialise_prior_distribution(prior_judgement)

                # Load data
                i_s.load_judgement_data(judgement_data, posterior_judgement, fit_judgement)

                internal_states.append(i_s)

            if len(internal_states) == 1:
                internal_states = internal_states[0]

            ## Action states
            action_states = []
            for model in action_states_list:
                a_s = models_dict['action'][model]['object'](N, K, 
                                                             *models_dict['action'][model]['params']['args'],
                                                             **models_dict['action'][model]['params']['kwargs'])
                # Load action data
                a_s.load_action_data(inters, inters_fit, data)

                action_states.append(a_s)

            if len(action_states) == 1: # Must be true atm, multiple action states are not supported
                action_states = action_states[0] 

            ## Sensory states
            sensory_states = []
            for model in sensory_states_list:
                sensory_s = models_dict['sensory'][model]['object'](N, K, 
                                                                    *models_dict['sensory'][model]['params']['args'],
                                                                    **models_dict['sensory'][model]['params']['kwargs'])

                sensory_states.append(sensory_s)
            
            if len(sensory_states) == 1: # Must be true atm, multiple sensory states are not supported
                sensory_states = sensory_states[0]

            # Create agent
            agent = Agent(N, sensory_states, internal_states, action_states)

            # Create experiment
            experiment = Experiment(agent, external_state)

            # Fit data
            experiment.fit()

            # Extract relevant data
            ## Populate a dataframe:
            ### Must happens for all internal states in internal states
            ### UID, pid, experiment, difficulty, scenario, model_name, log likelihood, prior_entropy, posterior_entropy, model_specs
            ## Collect posteriors for fitting
            utid = f'{part_experiment[-1]}_{participant}_{model_name}_{difficulty}'
            
            if isinstance(internal_states, list):
                for i, i_s in enumerate(internal_states):
                    # Posterior for each model
                    dfs_posteriors[utid] = i_s.posterior_over_models
                    if part_idx % 5 == 0 or part_idx == sample_size - 1:
                        dfs_posteriors(f'./data/model_fitting_outputs/{internal_states_list[i]}/posteriors.csv')


                    # Posteriors for each judgement and each trial
                    ## Set up posterior dataframe for each trial and for each model
                    ## Stored in a special folder (TBD...)
                    judge_trial_idx, judge_link_idx = np.where(judgement_data == True)
                    df_posteriors_trial = pd.DataFrame(columns=links, data=space_triple[0])
                    for j, j_idx in enumerate(judge_trial_idx):
                        links_value = links[judge_link_idx[j]]
                        # Add judgements
                        df_posteriors_trial[f'judgement_{j_idx}'] = df_posteriors_trial[f'links_{judge_link_idx[j]+1}'] == links_value
                        # Add posteriors
                        df_posteriors_trial[f'posterior_{j_idx}'] = i_s.posterior_over_models_byidx(j_idx)
                    
                    # Save df directly
                    df_posteriors_trial.to_csv(f'./data/model_fitting_output/{internal_states_list[i]}/trials/{internal_states_list[i]}_{utid}.csv')
                    
                    ## Generate dataframe
                    output = [
                        utid,
                        participant,
                        experiment,
                        difficulty,
                        model_name,
                        internal_states_list[i], # model name 
                        ground_truth, # ground truth model
                        i_s.MAP, # Posterior map
                        posterior_judgement,
                        prior_judgement,
                        i_s.prior_entropy, # Prior entropy
                        i_s.posterior_entropy, # Posterior entropy
                        None # Model specs, for specific parametrisations (None if irrelevant)
                    ]
                    out_df = pd.DataFrame(columns=df.columns, data=output)

                    df.append(out_df, ignore_index=True)

                    
            else:
                # Posterior for each model
                dfs_posteriors[utid] = i_s.posterior_over_models
                if part_idx % 5 == 0 or part_idx == sample_size - 1:
                    dfs_posteriors(f'./data/model_fitting_outputs/{internal_states_list[0]}/posteriors.csv')

                # Posteriors for each judgement and each trial
                ## Set up posterior dataframe for each trial and for each model
                ## Stored in a special folder (TBD...)
                judge_trial_idx, judge_link_idx = np.where(judgement_data == True)
                df_posteriors_trial = pd.DataFrame(columns=links, data=space_triple[0])
                for j, j_idx in enumerate(judge_trial_idx):
                    links_value = links[judge_link_idx[j]]
                    # Add judgements
                    df_posteriors_trial[f'judgement_{j_idx}'] = df_posteriors_trial[f'links_{judge_link_idx[j]+1}'] == links_value
                    # Add posteriors
                    df_posteriors_trial[f'posterior_{j_idx}'] = internal_states.posterior_over_models_byidx(j_idx) 
                
                # Save df directly
                df_posteriors_trial.to_csv(f'./data/model_fitting_output/{internal_states_list[0]}/trials/{internal_states_list[0]}_{utid}.csv')
                    
                output = [
                    utid,
                    participant,
                    experiment,
                    difficulty,
                    model_name,
                    internal_states_list[0],
                    ground_truth,
                    internal_states.MAP,
                    posterior_judgement,
                    prior_judgement,
                    internal_states.prior_entropy,
                    internal_states.posterior_entropy,
                    None, # Model specs, for specific parametrisations (None if irrelevant)
                ]
                out_df = pd.DataFrame(columns=df.columns, data=output)

                df.append(out_df, ignore_index=True)
                    


        if part_idx % 5 == 0:
            df.to_csv('./data/model_fitting_outputs/summary_data.csv', sep=';')
        
        part_idx += 1 

    # Final save
    df.to_csv('./data/model_fitting_outputs/dataframe.csv', sep=';')

    return df
        
            


# Create different version depending on the chosen free parameters set 
## Cannot be multiple internal states at once here

def fit_params(to_fit, *args):
    log_likelihood = fit_models(to_fit, *args)
    
    return - log_likelihood

