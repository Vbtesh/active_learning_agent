from methods.model_fitting_wrappers import fit_participant_param_wrapper


experiments = ['experiment_1', 'experiment_2', 'experiment_3']

data_path = [
    f'/mnt/c/Users/vbtes/CompProjects/vbtCogSci/csl_global_analysis/data/global_modelling_data.obj'
]

internal_states = [
    'change_obs_fk',
    'LC_discrete_att',
    'change_obs_fk',
    'LC_discrete_att'
]

outdir_path = [
    f'./data/params_fitting_outputs/'
]

tags = [
    ['att', 'cha'],
    ['att'],
    ['att', 'cha', 'prior'],
    ['att', 'prior']
]

for i in range(len(internal_states)):
    fit_participant_param_wrapper(data_path[0],
                                  experiments,
                                  internal_states[i],
                                  tags[i],
                                  outdir_path[0] + internal_states[i])
    print()