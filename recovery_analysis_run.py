from methods.model_fitting_wrappers import fit_participant_param_wrapper

# Must do: 
# ces : ONGOING
# change_d_obs_fk : ONGOING
# LC_discrete: ONGOING
# LC_discrete_attention: ONGOING
# normative: ONGOING

case = 'real_actions'

model = 'LC_discrete'
outdir_path = f'./data/recovery_analysis/{case}/{model}'

experiments = ['experiment_1', 'experiment_2', 'experiment_3']

data_path = [
    f'./data/recovery_analysis/{case}/_datasets/modelling_data_ces_strength_&_str_guess.obj',
    f'./data/recovery_analysis/{case}/_datasets/modelling_data_change_d_obs_fk_&_att_cha.obj',
    f'./data/recovery_analysis/{case}/_datasets/modelling_data_LC_discrete_&_1.obj',
    f'./data/recovery_analysis/{case}/_datasets/modelling_data_LC_discrete_attention_&_att.obj',
    f'./data/recovery_analysis/{case}/_datasets/modelling_data_normative_&_1.obj'
]

internal_states = [
    'ces_strength',
    'change_d_obs_fk',
    'LC_discrete',
    'LC_discrete_attention',
    'normative'
]

tags = [
    ['str', 'guess'],
    ['att', 'cha'],
    [],
    ['att'],
    []
]

for i in range(len(internal_states)):
    print(f'Fitting {internal_states[i]} to {model} data.')
    fit_participant_param_wrapper(data_path[i],
                                  experiments,
                                  internal_states[i],
                                  tags[i],
                                  outdir_path)
    print()