import numpy as np
import pickle

from classes.ou_network import OU_Network
from methods.sample_space_methods import build_space
from methods.model_fitting_utilities import fit_params_models_partlevel
from methods.states_params_importer import import_states_params_asdict

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


models_to_fit = [
    'LC_discrete_attention', # OK
    'change_d_obs_fk', # OK
    'change_d_obs_cause_effect',
    'change_d_obs_cause',
    'LC_discrete',
    'normative'
]
# Pick states to fit
internal_states_list = ['normative']
action_states_list = ['experience_vao']
sensory_states_list = ['omniscient']

fitting_softmax = True

if internal_states_list[0][:6] == 'change' and fitting_softmax:
    params_initial_guesses = [2/3, 0, 1/2]           
    params_bounds = [(0, 1), (0, 2000), (1/10, 1)]                       
    internal_params_labels = [['decay_rate', 0], ['smoothing', 1]]            
    action_params_labels = []              
    sensory_params_labels = [['change_memory', 2]]

elif internal_states_list[0][:6] == 'change' and not fitting_softmax:
    params_initial_guesses = [2/3, 1/2]           
    params_bounds = [(0, 1), (1/10, 1)]                       
    internal_params_labels = [['decay_rate', 0]]            
    action_params_labels = []              
    sensory_params_labels = [['change_memory', 1]]

elif internal_states_list[0] == 'LC_discrete_attention':
    params_initial_guesses = [2/3, 0]           
    params_bounds = [(0, 1), (0, 2000)]                       
    internal_params_labels = [['decay_rate', 0], ['smoothing', 1]]            
    action_params_labels = []               
    sensory_params_labels = []

elif internal_states_list[0] == 'normative' or internal_states_list[0] == 'LC_discrete':
    params_initial_guesses = [0]           
    params_bounds = [(0, 2000)]                       
    internal_params_labels = [['smoothing', 0]]            
    action_params_labels = []              
    sensory_params_labels = []

print(f'Fitting: {internal_states_list[0]}...')
print(f'Parameters: {internal_params_labels + action_params_labels + sensory_params_labels}')

# Import model dicts
models_dict = import_states_params_asdict()

# /!\ Data loss warning /!\
reset_summary = False
reset_posteriors = False
# /!\ Data loss warning /!\

# Run fitting function
summary = fit_params_models_partlevel(params_initial_guesses,
                                      params_bounds,
                                      internal_params_labels,
                                      action_params_labels,
                                      sensory_params_labels,
                                      internal_states_list,
                                      action_states_list,
                                      sensory_states_list,
                                      models_dict,
                                      selected_data)