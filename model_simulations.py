from matplotlib.font_manager import json_load
from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import pickle
import json
from os.path import exists
from datetime import datetime
import gc

from methods.simulation_utilities import simulate
from methods.metrics import normalised_euclidean_distance, find_indirect_errors
from methods.sample_space_methods import build_space_env, construct_link_matrix



"""
Set for controlled action or generated action
"""

part_data = True
part_action = True

# If part data is used, then so must actions
if part_data:
    part_action = True


if not part_action:
    ground_truth = np.array([1, 0, 0, 1, 0, 0])
    num_simulations = 50
    N = 200
else:
    # Get all chain models
    with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
        modelling_data = pickle.load(inFile)


    ## Data selection 
    ## Currently hard coded

    trials_data = []
    for k, v in modelling_data.items():
        exp = v['experiment']

        trials = [v['trials'][trial] for trial in v['trials'].keys()]
        #chain = [trial for trial in generic_trials if 'chain' in trial['name']]
        #if len(chain) > 0:
        #    chain = chain[0]
        #    chains_data.append(chain)
        #    chains_data[-1]['experiment'] = exp
        for trial in trials:
            trials_data.append(trial)

    
    num_simulations = len(trials_data)

print(f'Num simulations: {num_simulations}')

"""
Tests different combinations of parametrisations, graphs and/or external states parameter values for a set number of simulations
"""
 
# Build internal sample space


K = 3
links = np.array([-1, -0.5, 0, 0.5, 1])

space_triple = build_space_env(K, links)

theta = np.array([0.5], dtype=float)
evidence_weights = np.array([1],dtype=float)
#evidence_weights = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], dtype=float)

save_step = 25

link_matrix = construct_link_matrix(K)
link_vec = link_matrix[~np.eye(K, dtype=bool)]



external_state = 'OU_Network'
external_state_parameters = {
    'OU_Network': {
        'theta': theta,
        'sigma': np.array([3])
    }
}
internal_state = [
    #'mean_field_vis'
    'normative_&_1',
    #'LC_discrete_&_1',
    #'LC_discrete_att_&_att'
]
internal_state = internal_state[0]

internal_state_parameters = {
    'mean_field_vis': {
        'update_schedule': np.array([
            'omniscient',
            #'full', 
            #'single_variable'
        ]),
        'factorisation': np.array(['normative', 'local_computations']),
        'certainty_threshold': np.array([0.4], dtype=float),
        'evidence_weight': evidence_weights
    },
    'LC_discrete_&_1': {
        'evidence_weight': evidence_weights
    },
    'normative_&_1': {
        'evidence_weight': evidence_weights
    },
    'LC_discrete_att_&_att': {
        'decay_rate': np.array([0], dtype=float),
        'evidence_weight': evidence_weights
    }
}

param_type = [external_state for _ in external_state_parameters[external_state].keys()] + [internal_state for _ in internal_state_parameters[internal_state].keys()]
param_names = [param for param in external_state_parameters[external_state].keys()] + [param for param in internal_state_parameters[internal_state].keys()]
param_values = [param for param in external_state_parameters[external_state].values()] + [param for param in internal_state_parameters[internal_state].values()]
num_combinations = np.array([len(v) for v in external_state_parameters.values()]).prod() * np.array([len(v) for v in internal_state_parameters.values()]).prod()

total_simulations = num_simulations * num_combinations

print(total_simulations)


param_combinations = np.array([np.array(comb) for comb in itertools.product(*param_values)])

param_dicts = []

for v in range(param_combinations.shape[0]):
    param_dict = {
        external_state: {},
        internal_state: {}
    }

    for i in range(param_combinations.shape[1]):
        param_dict[param_type[i]][param_names[i]] = param_combinations[v, i]

    param_dicts.append(param_dict)


"""
Set up data storage variables

Set up file name
"""


# Define output file names
file_name = f'{internal_state}_sims'
file_tag = datetime.today().strftime('%Y-%m-%d')

full_file_name = f'./data/simulations_outputs/{file_name}_action{part_action}_data{part_data}_{file_tag}.csv'

"""
Set up dataframe

df is baseline
dict as obj for more complex data storage
"""

if part_action:
    id_cols = ['utid', 'uid', 'experiment', 'sign', 'trial_name', 'judgement', 'judgement_accuracy', 'judgement_to_MAP', 'judgement_to_weighted_MAP', 'part_theta', 'part_indirect_errors']
else:
    id_cols = ['sim_num']

external_states_cols = (np.array(param_names)[np.array(param_type) == external_state]).tolist()
internal_state_cols = ['internal_state'] + (np.array(param_names)[np.array(param_type) == internal_state]).tolist()
columns = ['sim_id'] + id_cols + external_states_cols + internal_state_cols + ['ground_truth', 'map_graph', 'accuracy', 'map_graph_weighted', 'accuracy_weighted', 'model_indirect_errors'] 
variational_cols = ['theta_MAP', 'sigma_MAP', 'theta_MAP_weighted', 'sigma_MAP_weighted', 'converged']
entropy_cols = link_vec
variational_entropy_cols = ['theta_H', 'sigma_H']
indirect_links_cols = ['num_indirect_links', 'indirect_links_loc']

columns += variational_cols + list(entropy_cols) + variational_entropy_cols + indirect_links_cols

df = pd.DataFrame(columns=columns)

if not exists(full_file_name):
    df.to_csv(full_file_name, index=False)
else:
    df = pd.read_csv(full_file_name)


"""
Run simulations
"""
for i, comb in enumerate(param_dicts):

    print(f'{i+1} out of {param_combinations.shape[0]}')
    for p in range(param_combinations.shape[1]): 
        print(f'{param_type[p]}, {param_names[p]}={param_combinations[i, p]}')


    #print(df.sim_id)
    for j in range(num_simulations):
        print(f'Progress: {np.round(j / num_simulations, 2)}')
        
        
        if part_action:
            utid = trials_data[j]['utid']
            # Sim_id
            es_string = "-es_" + "_".join(param_combinations[i, np.array(param_type) == external_state].astype(str).tolist())
            is_string = f"-is_{internal_state}_"  + "_".join(param_combinations[i, np.array(param_type) == internal_state].astype(str).tolist())
            sim_id = utid + es_string + is_string

            if sim_id in df.sim_id.to_list():
                print(f'Passing {sim_id}...')
                continue

        else:
            es_string = "-es_" + "_".join(param_combinations[i, np.array(param_type) == external_state].astype(str).tolist())
            is_string = f"-is_{internal_state}_"  + "_".join(param_combinations[i, np.array(param_type) == internal_state].tolist())
            sim_id = [str(j+1)] + es_string + is_string

            if sim_id in df.sim_id.to_list():
                print(f'Passing {sim_id}...')
                continue


        e_s_params = {external_state: comb[external_state]}
        i_s_params = {internal_state.split('_&_')[0]: comb[internal_state]}

        # Build experiment 
        if part_action:
            action_plan = 'data'
            data = trials_data[j]
            N = data['data'].shape[0]
            ground_truth = data['ground_truth']
        else:
            action_plan = None
            data = None

        experiment = simulate(N, K, ground_truth, space_triple,
                              external_state_params=e_s_params, 
                              internal_state_params=i_s_params,
                              realised_data=part_data,
                              use_action_plan=action_plan,
                              trial_data=data)
        
        # Collect data
        internal_state_done = experiment.agent.internal_state
        MAP_graph = experiment.agent.internal_state.MAP_unsmoothed
        MAP_graph_weighted = experiment.agent.internal_state.MAP_weighted
        new_index = df.index.size

        # Indirect errors
        errors, indirect_links_loc, out = find_indirect_errors(ground_truth, MAP_graph)
        if len(errors) > 0:
            model_indirect_errors = errors.sum()
        else:
            model_indirect_errors = 0
        if indirect_links_loc.sum() > 0:
            num_indirect_links = indirect_links_loc.sum()
        else:
            num_indirect_links = 0

        
        df.loc[new_index, 'sim_id'] = sim_id
        if part_action:
            judgement = trials_data[j]['posterior']
            if 'crime_control' in trials_data[j]['name']:
                sign = 'control'
                name = 'crime'
            elif len(trials_data[j]['name'].split('_')) == 3:
                sign, name, _ = trials_data[j]['name'].split('_')
            else:
                name, _ = trials_data[j]['name'].split('_')
                sign = None

            df.loc[new_index, 'utid'] = utid
            df.loc[new_index, 'experiment'] = int(utid[0])
            df.loc[new_index, 'sign'] = sign
            df.loc[new_index, 'trial_name'] = name
            df.loc[new_index, 'judgement'] = np.array2string(judgement)
            df.loc[new_index, 'judgement_accuracy'] = normalised_euclidean_distance(ground_truth, judgement)
            df.loc[new_index, 'judgement_to_MAP'] = normalised_euclidean_distance(judgement, MAP_graph)
            df.loc[new_index, 'judgement_to_weighted_MAP'] = normalised_euclidean_distance(judgement, MAP_graph_weighted)
            df.loc[new_index, 'part_theta'] = 0.5

            errors, indirect_links_loc, out = find_indirect_errors(ground_truth, MAP_graph)
            if len(errors) > 0:
                part_indirect_errors = errors.sum()
            else:
                part_indirect_errors = 0
            df.loc[new_index, 'part_indirect_errors'] = part_indirect_errors
        else:
            df.loc[new_index, 'sim_num'] = j+1
        df.loc[new_index, 'internal_state'] = internal_state
        df.loc[new_index, param_names] = param_combinations[i, :]
        df.loc[new_index, 'ground_truth'] = np.array2string(ground_truth)
        df.loc[new_index, 'map_graph'] = np.array2string(MAP_graph)
        df.loc[new_index, 'accuracy'] = normalised_euclidean_distance(ground_truth, MAP_graph)
        df.loc[new_index, 'map_graph_weighted'] = np.array2string(MAP_graph_weighted)
        df.loc[new_index, 'accuracy_weighted'] = normalised_euclidean_distance(ground_truth, MAP_graph_weighted)
        df.loc[new_index, 'model_indirect_errors'] = model_indirect_errors
        
        df.loc[new_index, entropy_cols] = internal_state_done.posterior_entropy_over_links_unsmoothed

        if internal_state_done.variational:
            df.loc[new_index, variational_cols] = list(internal_state_done.variational_MAP[0]) + list(internal_state_done.variational_MAP_weighted[0]) + [internal_state_done.converged]

            link_bool = internal_state_done._link_params_bool
            
            df.loc[new_index, variational_entropy_cols] = internal_state_done.variational_posterior_entropy[~link_bool]
        
        df.loc[new_index, 'num_indirect_links'] = num_indirect_links
        df.loc[new_index, 'indirect_links_loc'] = np.array2string(indirect_links_loc)
        ## Input parameters
        ## Ground truth
        ## Variational posterior
        ## Variatioanl entropy
        ## MAP parameters
        ## Accuracy

        del internal_state_done
        del experiment
        gc.collect()

        if j % save_step == 0:
            print('Saving...')
            df_old = pd.read_csv(full_file_name)
            pd.concat([df_old, df], ignore_index=True).to_csv(full_file_name, index=False)
            # Resets dfs
            df_old = None
            df = pd.DataFrame(columns=columns)

        
    print('Saving...')
    df_old = pd.read_csv(full_file_name)
    pd.concat([df_old, df], ignore_index=True).to_csv(full_file_name, index=False)
    # Resets dfs
    df_old = None
    df = pd.DataFrame(columns=columns)

        #print(gc.get_objects())

        


