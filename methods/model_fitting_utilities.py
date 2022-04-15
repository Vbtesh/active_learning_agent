from sys import intern
import numpy as np
import pandas as pd
import pickle
from os.path import exists

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
               fit_judgement=False,                 # List of internal states names as strings
               reset_summary=False,                 # /!\ Data loss warning /!\ if true resets the summary, else, append to it 
               reset_posteriors=False,              # /!\ Data loss warning /!\ if true reset the posteriors, else, append to it
               save_data=True,                      # /!\ Data miss warning /!\ if False does not save the results but simply fit experiments
               console=False):                      # If true, log progress on the console
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

    # If save data, generate frames
    if save_data:
        # Define DataFrame
        cols = ['utid', 'pid', 'experiment', 'difficulty', 'scenario', 'model_name', 'ground_truth', 'posterior_map', 'posterior_judgement', 'prior_judgement', 'prior_entropy', 'posterior_entropy', 'model_specs']
        df = pd.DataFrame(columns=cols)
        ## If summary data does not exist, create it
        if not exists('./data/model_fitting_outputs/summary_data.csv') or reset_summary:
            pd.DataFrame(columns=cols).to_csv('./data/model_fitting_outputs/summary_data.csv', sep=';', index=False)


        # Posterior DataFrame
        # One per internal state
        links_cols = [f'link_{i}' for i in range(K**2 - K)]
        dfs_posteriors = []
        for model in internal_states_list:
            # Create files if they don't exist yet
            if not exists(f'./data/model_fitting_outputs/{model}/posteriors.csv') or reset_posteriors:
                dfs_p = pd.DataFrame(columns=links_cols, data=space_triple[0])
                dfs_p.to_csv((f'./data/model_fitting_outputs/{model}/posteriors.csv'), index=False)
            else:
                dfs_p = pd.read_csv(f'./data/model_fitting_outputs/{model}/posteriors.csv')

            # Stores index and initialise empty dataframe of posteriors
            posterior_index = dfs_p.index
            dfs_posteriors.append(pd.DataFrame(index=posterior_index))
            # Reset df
            dfs_p = None
        
    # Count participant index
    sample_size = len(data_dict.keys())
    part_idx = 0

    for participant, part_data in data_dict.items():
        print(f'participant {part_idx+1}, out of {sample_size}', )

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
            judgement_data = trial_data['links_hist'] # Change in judgement sliders
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
                                                               generate_sample_space = not build_space)

                # Initialse space according to build_space
                if build_space:
                    i_s.add_sample_space_env(space_triple)

                # Initialise prior distributions for all IS
                i_s.initialise_prior_distribution(prior_judgement)

                # Load data
                i_s.load_judgement_data(judgement_data, posterior_judgement, fit_judgement)

                internal_states.append(i_s)


            ## Action states
            action_states = []
            for model in action_states_list:
                a_s = models_dict['actions'][model]['object'](N, K, 
                                                             *models_dict['actions'][model]['params']['args'],
                                                             **models_dict['actions'][model]['params']['kwargs'])
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
            if len(internal_states) == 1:
                agent = Agent(N, sensory_states, internal_states[0], action_states)
            else:
                agent = Agent(N, sensory_states, internal_states, action_states)

            # Create experiment
            experiment = Experiment(agent, external_state)

            # Fit data
            experiment.fit(console=console)

            # If not saving data, continue here
            if not save_data:
                continue

            # Extract relevant data
            ## Populate a dataframe:
            ### Must happens for all internal states in internal states
            ### UID, pid, experiment, difficulty, scenario, model_name, log likelihood, prior_entropy, posterior_entropy, model_specs
            ## Collect posteriors for fitting
            utid = f'{part_experiment[-1]}_{participant}_{model_name}_{difficulty}'
 
            for i, i_s in enumerate(internal_states):
                # Posterior for each model
                dfs_posteriors[i][utid] = i_s.posterior_over_models

                # Every 5 participants, save data and reset dfs_posteriors
                if part_idx % 5 == 0 or part_idx == sample_size - 1:
                    df_old = pd.read_csv(f'./data/model_fitting_outputs/{internal_states_list[i]}/posteriors.csv')
                    pd.concat([df_old, dfs_posteriors[i]], axis=1).to_csv(f'./data/model_fitting_outputs/{internal_states_list[i]}/posteriors.csv', index=False)
                    # Reset dfs
                    df_old = None  
                    dfs_posteriors[i] = pd.DataFrame(index=posterior_index)
                
                # Posteriors for each judgement and each trial
                ## Set up posterior dataframe for each trial and for each model
                ## Stored in a special folder (TBD...)
                judge_trial_idx, judge_link_idx = np.where(judgement_data == True)
                df_posteriors_trial = pd.DataFrame(columns=links_cols, data=space_triple[0])

                for j, j_idx in enumerate(judge_trial_idx):
                    value = judgement_data[judge_trial_idx, judge_link_idx[j]][0]
                    # Add judgements
                    df_posteriors_trial[f'judgement_{j_idx}'] = df_posteriors_trial[f'link_{judge_link_idx[j]}'] == value
                    # Add posteriors
                    posterior_over_models = np.squeeze(i_s.posterior_over_models_byidx(j_idx))
                    df_posteriors_trial[f'posterior_{j_idx}'] = posterior_over_models
                
                # Save df directly
                df_posteriors_trial.to_csv(f'./data/model_fitting_outputs/{internal_states_list[i]}/trials/{internal_states_list[i]}_{utid}.csv', index=False)
                
                ## Generate summary dataframe entry
                output = [
                    utid,
                    participant,
                    part_experiment,
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

                data_output = {df.columns[i]:[output[i]] for i in range(len(df.columns))}
                out_df = pd.DataFrame(data=data_output)

                df = pd.concat([df, out_df])


        part_idx += 1 

        # If not saving data, continue here
        if not save_data:
            continue
        
        # Save data every 5 participants and reset df
        if part_idx % 5 == 0:
            df_old = pd.read_csv('./data/model_fitting_outputs/summary_data.csv', sep=';')
            pd.concat([df_old, df], ignore_index=True).to_csv('./data/model_fitting_outputs/summary_data.csv', sep=';', index=False)
            # Resets dfs
            df_old = None
            df = pd.DataFrame(columns=cols)

    
    # Final save
    if save_data:
        df_old = pd.read_csv('./data/model_fitting_outputs/summary_data.csv', sep=';')
        pd.concat([df_old, df], ignore_index=True).to_csv('./data/model_fitting_outputs/summary_data.csv', sep=';', index=False)
            

        return pd.read_csv('./data/model_fitting_outputs/summary_data.csv', sep=';')





# Recover final judgements
# These need to be recovered as a dataframe with links in the 6 first columns and then the index of the model for each columns
def extract_final_judgements(data_dict):      # Dict of data for each participants/trial/experiment

    # General loop
    ## Initialise general invariant parameters
    K = 3
    links = np.array([-1, -0.5, 0, 0.5, 1]) # Possible link values

    # Build internal sample space
    space_triple = build_space_env(K, links)
        
    # Count participant index
    sample_size = len(data_dict.keys())
    part_idx = 0

    # Posterior DataFrame
    # One per internal state
    links_cols = [f'link_{i}' for i in range(K**2 - K)]
    # Create files if they don't exist yet
    if not exists(f'./data/model_fitting_outputs/final_judgements.csv'):
        df_links = pd.DataFrame(columns=links_cols, data=space_triple[0])
        df_links.to_csv(f'./data/model_fitting_outputs/final_judgements.csv', index=False)
    else:
        df_links = pd.read_csv(f'./data/model_fitting_outputs/final_judgements.csv')

    # Stores index and initialise empty dataframe of posteriors
    posterior_index = df_links.index
    df = pd.DataFrame(index=posterior_index)
    

    for participant, part_data in data_dict.items():
        print(f'participant {part_idx+1}, out of {sample_size}', )

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
            judgement_data = trial_data['links_hist'] # Change in judgement sliders
            posterior_judgement = trial_data['posterior'] # Final states of judgement sliders
            prior_judgement = trial_data['prior'] if 'prior' in trial_data.keys() else None

            # Extract relevant data
        
            utid = f'{part_experiment[-1]}_{participant}_{model_name}_{difficulty}'

            final_judgement = (space_triple[0] == posterior_judgement).all(axis=1)

            # Posterior for each model
            df[utid] = final_judgement

            # Every 5 participants, save data and reset dfs_posteriors
            if part_idx % 30 == 0:
                df_old = pd.read_csv(f'./data/model_fitting_outputs/final_judgements.csv')
                pd.concat([df_old, df], axis=1).to_csv(f'./data/model_fitting_outputs/final_judgements.csv', index=False)
                # Reset dfs
                df_old = None  
                df = pd.DataFrame(index=posterior_index)


        if part_idx == sample_size - 1:
            df_old = pd.read_csv(f'./data/model_fitting_outputs/final_judgements.csv')
            pd.concat([df_old, df], axis=1).to_csv(f'./data/model_fitting_outputs/final_judgements.csv', index=False)

        part_idx += 1



## Softmax temperature log likelihood
def softmax_neg_log_likelihood(temp, dataset, selection):
    softmax_unnorm = np.exp(dataset * temp)
    softmax = softmax_unnorm / softmax_unnorm.sum(axis=0).reshape((1, dataset.shape[1]))

    judgements_likelihood = softmax[selection]

    log_likelihood = np.log(judgements_likelihood).sum()
    return - log_likelihood

