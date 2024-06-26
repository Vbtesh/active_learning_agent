from methods.model_fitting_wrappers import fit_group_param_wrapper


experiments = [
    ['experiment_1'], 
    ['experiment_2'], 
    ['experiment_3'],
    ['experiment_4']
]

data_path = [
    f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj'
]

internal_states = [
    'Adaptive_LC',
    'Adaptive_LC',
    #'Adaptive_Selective_LC',
    #'Adaptive_Selective_LC',
    #'LC_discrete_att_all',
    #'LC_discrete_att_all',
    #'LC_discrete_att_all',
    #'change_obs_fk',
    #'LC_discrete_att',
    #'change_obs_fk',
    #'LC_discrete_att',
    #'LC_discrete_att',
    #'LC_discrete', # Group 3
    #'LC_discrete', 
    #'ces_strength', # Group 4
    #'ces_no_strength',
    #'normative',
    #'normative' # Group 2
]

tags = [
    ['1'],
    ['prior'],
    ['1'],
    ['prior'],
    #['1'],
    #['prior'],
    #['att', 'prior'],
    #['att', 'cha'],
    #['att'],
    #['att', 'cha', 'prior'],
    #['att', 'prior'],
    #['1'],
    #['prior'],
    #['1'],
    #['prior'],
    #['str', 'guess'],
    #['guess'],
    #['1'],
    #['prior'] # Group 2
]

outdir_path = [
    f'./data/group_params_fitting_outputs/'
]

for e in range(len(experiments)):
    for i in range(len(internal_states)):
        fit_group_param_wrapper(data_path[0],
                                experiments[e],
                                internal_states[i],
                                tags[i],
                                outdir_path[0])
        print()