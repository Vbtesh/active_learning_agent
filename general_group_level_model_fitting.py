from methods.model_fitting_wrappers import fit_group_param_wrapper


experiments = [
    ['experiment_1'], 
    #['experiment_2'], 
    #['experiment_3']
]

data_path = [
    f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj'
]

internal_states = [
    #'change_obs_fk',
    #'LC_discrete_att',
    #'change_obs_fk',
    'LC_discrete_att',
    #'normative',
    #'normative', # Group 2
    #'LC_discrete', # Group 3
    #'LC_discrete', 
    #'ces_strength', # Group 4
    #'ces_no_strength'
]

outdir_path = [
    f'./data/group_params_fitting_outputs/'
]

tags = [
    #['att', 'cha'],
    #['att'],
    #['att', 'cha', 'prior'],
    ['att', 'prior'],
    #['1'],
    #['prior'], # Group 2
    #['1'],
    #['prior'],
    #['str', 'guess'],
    #['guess']
]

for e in range(len(experiments)):
    for i in range(len(internal_states)):
        fit_group_param_wrapper(data_path[0],
                                experiments[e],
                                internal_states[i],
                                tags[i],
                                outdir_path[0])
        print()