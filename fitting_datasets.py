from matplotlib.font_manager import json_load
import numpy as np
import pandas as pd
import pickle
import json

from methods.model_fitting_utilities import fit_models
from methods.states_params_importer import import_states_params_asdict


## Import behavioural experiment
with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
    modelling_data = pickle.load(inFile)

## 
experiments = ['experiment_1', 'experiment_2', 'experiment_3']

print(len(modelling_data.keys()))
selected_data = {}
pick_interval = 1
idx = 0
for part, data in modelling_data.items():
    if data['experiment'] in experiments and idx % pick_interval == 0:
        selected_data[part] = data
    
    idx += 1

print(len(selected_data.keys()))



models_dict = import_states_params_asdict()

use_fitted_parameters = True
if use_fitted_parameters:
    with open('./data/params_fitting_outputs/fitted_params.json', 'r') as infile:
        use_fitted_parameters = json.load(infile)

    internal_states_list = list(use_fitted_parameters[list(use_fitted_parameters.keys())[0]].keys())
    #internal_states_list = [model for model in internal_states_list if model[:3] == 'ces']
else:
    internal_states_list = ['change_obs_fk']
action_states_list = ['experience_vao']
sensory_states_list = ['omniscient' for _ in internal_states_list]

# Import model dicts
models_dict = import_states_params_asdict()

# /!\ Data loss warning /!\
save_data = True
# /!\ Data loss warning /!\
console = False

tag='_current'

print(f'File tag: {tag}')
# Fit models 
fit_models(internal_states_list,
           action_states_list,
           sensory_states_list,
           models_dict,
           selected_data,
           use_fitted_params=use_fitted_parameters,
           save_data=save_data,
           console=console,
           file_tag=tag)