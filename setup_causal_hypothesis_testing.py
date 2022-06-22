import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

def calc_accuracy(ground_truth_str, MAP_str):
    ground_truth = np.array(" ".join(ground_truth_str[1:-1].split(' ')).split(), dtype=float)
    MAP = np.array(" ".join(MAP_str[1:-1].split(' ')).split(), dtype=float)

    acc = 1-pdist(np.stack((ground_truth, MAP)))[0] / np.linalg.norm(abs(np.array(ground_truth)) + 2*np.ones((1, 6)))

    return acc

# Generate long form datasets to test possible causal hypotheses in R

locations = {
    'TA': './data/model_fitting_outputs/summary_data_fit_true.csv', # True Action (TA)
    'RA': './data/model_fitting_outputs/summary_data_run_random.csv', # Random actions (RA)
    'SA': './data/model_fitting_outputs/summary_data_run_inters_sparse.csv', # Sparse actions (SA)
    'CI': './data/model_fitting_outputs/summary_data_run_CI.csv' # congruence inverted: use congruent actions in incongruent trials and vice versa
    # GA: use their generic actions in labelled trials
    # FR: randomise the pool of labelled actions and realocate randomly
}

tag_dict = {
    'TA': 1,
    'CI': 2,
    'SA': 3,
    'RA': 4
}

models = [
    'normative_&_1',
    'normative_&_prior',
    'LC_discrete_&_1',
    'LC_discrete_&_prior',
    'LC_discrete_attention_&_att',
    'LC_discrete_attention_&_att_prior',
    'LC_discrete_att_&_att',
    'LC_discrete_att_&_att_prior',
    'change_d_obs_fk_&_att_cha',
    'change_d_obs_fk_&_att_cha_prior',
    'change_obs_fk_&_att_cha',
    'change_obs_fk_&_att_cha_prior',
    'ces_strength_&_str_guess',
    'ces_no_strength_&_guess'
]

cols = ['utid', 'pid', 'experiment', 'difficulty', 'scenario', 'actions', 'accuracy', 'judgement_accuracy', 'posterior_entropy']
df_all = pd.DataFrame()
for tag, loc in locations.items():

    df = pd.read_csv(loc)

    df['accuracy'] = df.apply(lambda x: calc_accuracy(x.ground_truth, x.posterior_map), axis=1)
    # Need to recover participants judgements here
    df['judgement_accuracy'] = df.apply(lambda x: calc_accuracy(x.posterior_judgement, x.posterior_map), axis=1)
    df['actions'] = tag
    
    for model in models:
        model_df = df[df.model_name == model][cols].rename({'pid': 'participant'}, axis=1)
        model_df = model_df[model_df.scenario.isin(['crime', 'estate', 'finance'])]
        diff_label = model_df.difficulty.to_list()
        model_df.difficulty = model_df.difficulty.replace({'congruent':1, 'incongruent':2, 'implausible':3})
        diff = model_df.difficulty.to_list()
        model_df[model_df.experiment == 'experiment_2'].to_csv(f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/simulated_data/{model}_2_{tag}.csv')
        model_df[model_df.experiment == 'experiment_3'].to_csv(f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/simulated_data/{model}_3_{tag}.csv')
        model_df[model_df.experiment == 'experiment_4'].to_csv(f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/simulated_data/{model}_4_{tag}.csv')


    df_all = pd.concat([df_all, df], axis=0)

# Model wise df of all action tags
for model in models:
    model_df = df_all[df_all.model_name == model][cols].rename({'pid': 'participant'}, axis=1)
    model_df = model_df[model_df.scenario.isin(['crime', 'estate', 'finance'])]
    model_df.difficulty = model_df.difficulty.replace({'congruent':1, 'incongruent':2, 'implausible':3})
    model_df.actions = model_df.actions.replace(tag_dict)

    model_df[model_df.experiment == 'experiment_2'].to_csv(f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/simulated_data/{model}_2_all.csv')
    model_df[model_df.experiment == 'experiment_3'].to_csv(f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/simulated_data/{model}_3_all.csv')
    model_df[model_df.experiment == 'experiment_4'].to_csv(f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/simulated_data/{model}_4_all.csv')


    



