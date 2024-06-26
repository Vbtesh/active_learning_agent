from matplotlib.font_manager import json_load
import numpy as np
import pandas as pd
import pickle
import json

from methods.model_fitting_utilities import fit_models
from methods.states_params_importer import import_states_params_asdict
from methods.action_plans import generate_action_plan


## Import behavioural experiment
with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
#with open('./data/.modelling_data_manipulated/congruence_inverted_CI.obj', 'rb') as inFile:
    modelling_data = pickle.load(inFile)

## 
experiments = [
    'experiment_1', 
    'experiment_2', 
    'experiment_3',
    'experiment_4'
]

exceptions = [
    '566feba6b937e400052d33b2', 
    '5f108dea719866356702d26f', 
    #'5fbfe145e52a44000a9c2966'
]

print(len(modelling_data.keys()))
selected_data = {}
pick_interval = 1
idx = 0
for part, data in modelling_data.items():
    #if part in exceptions:
    #    selected_data[part] = data
    #if data['experiment'] in experiments and idx % pick_interval == 0:
    #    selected_data[part] = data

    #if part not in exceptions:
    #    if data['experiment'] in experiments:
    #        selected_data[part] = data

    if part == '5fbfe145e52a44000a9c2966':
        selected_data[part] = data
    
    idx += 1

print(len(selected_data.keys()))


#for part, part_data in selected_data.items():
#    print(f'|_ {part}')
#    for m_type in part_data['trials'].keys():
#        print(f'    |_ {m_type}')


models_dict = import_states_params_asdict()

use_fitted_parameters = False

if use_fitted_parameters:
    with open('./data/params_fitting_outputs/fitted_params.json', 'r') as infile:
        use_fitted_parameters = json.load(infile)

    internal_states_list_full = list(use_fitted_parameters[list(use_fitted_parameters.keys())[0]].keys())
    internal_states_list = []
    for model in internal_states_list_full:
        if 'normative' in model:
            internal_states_list.append(model)
        if 'LC_discrete_att_' in model:
            internal_states_list.append(model)
        if 'change_obs_fk' in model:
            internal_states_list.append(model)
        if 'ces' in model:
            internal_states_list.append(model)
        if 'LC_discrete_&' in model:
            internal_states_list.append(model)
        if 'LC_discrete_att_all' in model:
            internal_states_list.append(model)
        if 'Adaptive' in model:
            internal_states_list.append(model)
else:
    internal_states_list = [
        #'normative_&_1',
        'LC_discrete_&_1',
        #'LC_discrete_att_&_att'
        #'Adaptive_Selective_LC_&_1'
    ]

action_states_list = [
    #'experience_vao',
    #'tree_search_soft_horizon',
    'tree_search_hard_horizon'
]
sensory_states_list = ['omniscient' for _ in internal_states_list]

# Import model dicts
models_dict = import_states_params_asdict()

# /!\ Data loss warning /!\
save_data = True
save_full_data = False
# /!\ Data loss warning /!\
verbose = False

# Action plan
# Possible plans
## Floats [0, 1]: will generate swipes between 90 and -90 for a total intervened time of the argument
## 'real_actions': will use the behavioural actions
## 'inverted_actions': NOT CODED YET
## 'actor': /!\ ONLY ONE INTERNAL STATE AT A TIME /!\ will use the action states specified in action_states_list 
## 'obs': will stay idle
## 'random': will pick random actions
use_action_plan = 'actor'

# Add tag at end of file
tag = '_OA_part'


print('CONFIGURATION')
print(f'Internal states: {internal_states_list}')
if use_action_plan == 'actor':
    internal_states_list = [internal_states_list[0]]
    tag = tag + '_' + internal_states_list[0]
    print(f'Action state: {use_action_plan}. Using {action_states_list[0]} action state, pairing with {internal_states_list[0]}...')
else:
    print(f'Action state: {use_action_plan}')
print(f'File tag: {tag}')
print(f'Save config: \n Save summary: {save_data} \n Save full posteriors and entropy: {save_full_data}')
print()


# Fit models 
fit_models(internal_states_list,
           action_states_list,
           sensory_states_list,
           models_dict,
           selected_data,
           fit_or_run='run',
           use_action_plan=use_action_plan,
           use_fitted_params=use_fitted_parameters,
           save_data=save_data,
           save_full_data=save_full_data,
           verbose=verbose,
           file_tag=tag)