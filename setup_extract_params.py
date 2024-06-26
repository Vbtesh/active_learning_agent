import numpy as np
import pandas as pd
from ast import literal_eval
import json
from os.path import exists

import matplotlib.pyplot as plt
import seaborn as sns

models_ran = [
    #'LC_discrete_attention',
    #'change_d_obs_fk',
    #'change_d_obs_cause_effect',
    #'change_d_obs_cause',
    'LC_discrete',
    'normative',
    'ces_strength',
    'ces_no_strength',
#    'LC_discrete_att',
#    'LC_discrete_att_all',
    'Adaptive_LC',
    'Adaptive_Selective_LC',
    'change_obs_fk'
]

file_tags = [
    #['att', 'att_prior'],
    #['att_cha', 'att_cha_prior'],
    #['att_cha'],
    #['att_cha'],
    [1, 'prior'],
    [1, 'prior'],
    ['str_guess'],
    ['guess'],
    #['att', 'att_prior'],
    #['att', 'att_prior'],
    [1, 'prior'],
    [1, 'prior'],
    ['att_cha', 'att_cha_prior']
]

model_labels = [
    #['LC w. attention', 'LC w. attention w. prior'],
    #['Change w. full knowledge', 'Change w. full knowledge w. prior'],
    #['Change linear cause effect'],
    #['Change linear cause'],
    ['LC basic', 'LC basic w. prior'],
    ['normative', 'normative w. prior'],
    ['CES strength sensitive'],
    ['CES basic'],
    #['Adaptive LC att.', 'Adaptive LC att. w. prior'],
    #['AS LC att.', 'AS LC att. w. prior'],
    ['Adaptive LC', 'Adaptive LC w. prior'],
    ['AS LC', 'AS LC w. prior'],
    ['Change', 'Change w. prior']
]

model_names = []
for names in model_labels:
    model_names += names

# Import datasets
df = pd.DataFrame()
for i, model in enumerate(models_ran):
    for j, tag in enumerate(file_tags[i]):

        if exists(f'./data/params_fitting_outputs/{model}/summary_fit_{tag}.csv'):
            good_path = f'./data/params_fitting_outputs/{model}/summary_fit_{tag}.csv'
        elif exists(f'./data/params_fitting_outputs/{model}/exp1234_{model}_&_{tag}.csv'):
            good_path = f'./data/params_fitting_outputs/{model}/exp1234_{model}_&_{tag}.csv'
        else:
            good_path = f'./data/params_fitting_outputs/{model}/exp123_{model}_&_{tag}.csv'

        if df.empty:        
            df = pd.read_csv(good_path)
            df = df.replace([model], [model_labels[i][j]])
            df['tag'] = tag
            df['folder'] = model
    
        else:
            df_2 = pd.read_csv(good_path)
            df_2 = df_2.replace([model], [model_labels[i][j]])
            df_2['tag'] = tag
            df_2['folder'] = model
            df = pd.concat([df, df_2], axis=0, ignore_index=True)


experiments_series = df_2.experiment.to_list()

pids = df.pid.unique()
df['best_fit'] = np.nan

for pid in pids:
    df_pid = df[df.pid == pid].sort_values('bic')
    df.loc[df.pid == pid, 'best_fit'] = df_pid.model_name.to_list()[0]

df_bic = pd.DataFrame(index=df.pid.unique(), columns=model_names)
for model in model_names:
    df_bic[model] = df[df.model_name == model].bic.to_list()

df_bic = df_bic[df_bic.mean().sort_values().index]
df_bic['Baseline'] = -2 * 4 * np.log(1/5**6)
df_bic.loc[df[df.experiment == 'experiment_4'].pid.unique(), 'Baseline'] = -2 * 5 * np.log(1/5**6)
df_bic['experiment'] = experiments_series


select_lc_attention = [
    'AS LC',
    'AS LC w. prior',
    'Baseline'
]
select_change = [
    'Change',
    'Change w. prior',
    'Baseline'
]

df_prior = pd.DataFrame(index=pids, columns=['experiment', 'pid', 'lc_bic', 'lc_prior', 'change_bic', 'change_prior'])

df_prior['experiment'] = experiments_series
df_prior['pid'] = df_2.pid.to_list()
df_prior['lc_bic'] = df[df.model_name == 'AS LC w. prior'].bic.to_list()
df_prior['change_bic'] = df[df.model_name == 'Change w. prior'].bic.to_list()

df_prior['lc_prior'] = df[df.model_name == 'AS LC w. prior'].apply(lambda x: float(" ".join(x.params[1:-1].split()).split(' ')[-1]), axis=1).to_list()
df_prior['change_prior'] = df[df.model_name == 'Change w. prior'].apply(lambda x: float(" ".join(x.params[1:-1].split()).split(' ')[-1]), axis=1).to_list()

df_prior['best_change'] = df_bic[select_change].apply(lambda x: np.argmin(x), axis=1)
#df_prior['best_change'] = df_prior['best_change'].replace(np.arange(len(select_change)), select_change)

df_prior['best_lc'] = df_bic[select_lc_attention].apply(lambda x: np.argmin(x), axis=1)
#df_prior['best_lc'] = df_prior['best_lc'].replace(np.arange(len(select_lc_attention)), select_lc_attention)
df_prior.to_csv('/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/prior_fitting_data.csv', index=False)


fitted_params_dict = {}

for i, model in enumerate(model_names):
    
    df_model = df[df.model_name == model]
    df_params = pd.DataFrame(index=df_model.pid)
    #df_params['experiment'] = df_model.experiment.to_list()

    folder = df_model.loc[df_model.index[0], 'folder']
    tag = df_model.loc[df_model.index[0], 'tag']

    params, indices = zip(*literal_eval(df_model.loc[df_model.index[0], 'params_labels']))
    columns = []
    for j in range(len(params)):
        columns.append(params[indices.index(j)])

    for j, col in enumerate(columns): 
        df_params[col] = df_model.apply(lambda x: float(" ".join(x.params[1:-1].split()).split(' ')[j]), axis=1).to_list()

    
    for j, pid in enumerate(df_params.index):
        if i == 0:
            fitted_params_dict[pid] = {}

        part_data = df_params.loc[pid].to_dict()
        fitted_params_dict[pid][folder +'_&_'+ str(tag)] = part_data
    #df_params.to_csv(f'./data/params_fitting_outputs/{folder}/parameters_{tag}.csv')

with open('./data/params_fitting_outputs/fitted_params.json', 'w') as outfile:
    json.dump(fitted_params_dict, outfile)


#fitted_params_dict