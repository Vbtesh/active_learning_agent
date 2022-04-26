import numpy as np
import pandas as pd
import pickle

from classes.ou_network import OU_Network
from methods.sample_space_methods import build_space
from methods.model_fitting_utilities import fit_models
from methods.states_params_importer import import_states_asdict, import_states_params_asdict


## Import behavioural experiment
with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
    modelling_data = pickle.load(inFile)

experiments = ['experiment_3']

print(len(modelling_data.keys()))
selected_data = {}
pick_interval = 28
idx = 0
for part, data in modelling_data.items():
    if data['experiment'] in experiments and idx % pick_interval == 0:
        selected_data[part] = data
    
    idx += 1

print(len(selected_data.keys()))


for part, part_data in selected_data.items():
    print(f'|_ {part}')
    for m_type in part_data['trials'].keys():
        print(f'    |_ {m_type}')


states_dict = import_states_asdict()
models_dict = import_states_params_asdict()


internal_states_list = ['change_d_obs_fk']
action_states_list = ['experience_vao']
sensory_states_list = ['omniscient']
external_state = OU_Network


# /!\ Data loss warning /!\
save_data = False
# /!\ Data loss warning /!\
console = True


# Fit models 
fit_models(internal_states_list,
           action_states_list,
           sensory_states_list,
           models_dict,
           selected_data,
           save_data=save_data,
           console=console)