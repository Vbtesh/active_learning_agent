import numpy as np
import pickle

from classes.ou_network import OU_Network
from methods.sample_space_methods import build_space
from methods.model_fitting_utilities import extract_final_judgements
from methods.states_params_importer import import_states_asdict, import_params_asdict, import_states_params_asdict

## Import behavioural experiment
with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
    modelling_data = pickle.load(inFile)

# Choose experiments
experiments = ['experiment_1', 'experiment_2', 'experiment_3']

# Select participans
selected_data = {}
pick_interval = 1
idx = 0
for part, data in modelling_data.items():
    if data['experiment'] in experiments and idx % pick_interval == 0:
        selected_data[part] = data
    
    idx += 1

# Extract judgements
extract_final_judgements(selected_data)
