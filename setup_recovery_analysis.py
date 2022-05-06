import pickle
import pandas as pd
import numpy as np
import json


def str_to_arr(model_str):
    return np.array(" ".join(model_str[1:-1].split(' ')).split(), dtype=float)

# Generate specific modelling data datasets

# Fit models (part_level) to each dataset
## fit_participant_params:
### build wrapper for outfile path and name
models_to_recover = [
    'normative_&_1',
    'LC_discrete_&_1',
    'LC_discrete_attention_&_att',
    'change_d_obs_fk_&_att_cha',
    'ces_strength_&_str_guess'
]

case = 'controlled_actions'
# Import model predictions for the current set of optimal parameters
df = pd.read_csv('./data/model_fitting_outputs/summary_data_run_inters.csv')

## Import behavioural experiment
with open('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj', 'rb') as inFile:
    modelling_data = pickle.load(inFile)


for model in models_to_recover:
    file_name = f'modelling_data_{model}.obj'

    df_model = df[df.model_name == model]

    for part, data in modelling_data.items():
        for trial, content in data['trials'].items():
            utid = f"{data['experiment'][-1]}_{part}_{content['name'][:-2]}_{trial}"
            posterior_map = str_to_arr(df_model.loc[df_model.utid == utid, 'posterior_map'].values[0])
            modelling_data[part]['trials'][trial]['posterior'] = posterior_map

    with open(f'./data/recovery_analysis/{case}/_datasets/{file_name}', 'wb') as outFile:
        pickle.dump(modelling_data, outFile)
