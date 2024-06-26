import pickle

from methods.model_fitting_utilities import fit_params_models_partlevel, fit_params_models_grouplevel
from methods.states_params_importer import import_states_params_asdict, params_to_fit_importer

"""
High level function to fit parameters to participants
Inputs:
    data_path: path to input data file (.obj containing a dictionary)
    experiments: experiments selected (as list of strings)
    internal_state: internal state name
    parameter_tags: parameter tags (as list of tags or string of tags, see below for possibilities)
    outdir_path: path to output data file (absolute path to target directory)

Output:
    pd.DataFrame: df containing fitting summary for the specified data, internal state and tags
"""

def fit_participant_param_wrapper(data_path,
                                  experiments,
                                  internal_state,
                                  parameter_tags,
                                  outdir_path):
    
    exp_str = 'exp' + ''.join([experiment[-1] for experiment in experiments])
    
    # Define outfile path
    if outdir_path[-4:] == 'csv':
        outfile = outdir_path
    else:
        outfile = f'{outdir_path}/{exp_str}_{internal_state}_&_{"_".join(parameter_tags)}.csv'
    
    ## Import behavioural experiment
    with open(data_path, 'rb') as inFile:
        modelling_data = pickle.load(inFile)

    # Select participans
    selected_data = {}
    pick_interval = 1
    idx = 0
    for part, data in modelling_data.items():
        if data['experiment'] in experiments and idx % pick_interval == 0:
            selected_data[part] = data

        idx += 1

    # Pick states to fit
    internal_states_list = [internal_state]
    action_states_list = ['experience_vao']
    sensory_states_list = ['omniscient']

    fitting_change = False
    fitting_attention = False
    if 'change' in internal_state:
        if 'cha' in parameter_tags:
            fitting_change = True          
        
        if 'att' in parameter_tags:
            fitting_attention = True 


    if 'LC_discrete' in internal_state:
        if 'att' in parameter_tags:
            fitting_attention = True 
            

    # CES
    fitting_guess = False
    fitting_strength = False
    if internal_state[:3] == 'ces':
        if 'guess' in parameter_tags:
            fitting_guess = True 

        if 'str' in parameter_tags:
            fitting_strength = True

    # Prior
    fitting_prior = False
    if 'prior' in parameter_tags:
        fitting_prior = True
   
    random_increment = 1

    params_to_fit_tuple = params_to_fit_importer(internal_states_list[0], 
                                                 fitting_change=fitting_change,
                                                 fitting_attention=fitting_attention,
                                                 fitting_guess=fitting_guess,
                                                 fitting_strength=fitting_strength,
                                                 fitting_prior=fitting_prior,
                                                 random_increment=random_increment)
    params_initial_guesses = params_to_fit_tuple[0]
    params_bounds = params_to_fit_tuple[1]
    internal_params_labels = params_to_fit_tuple[2]
    action_params_labels = params_to_fit_tuple[3]
    sensory_params_labels = params_to_fit_tuple[4]
    fitting_list = params_to_fit_tuple[5]


    print(f'Fitting: {internal_states_list[0]}...')
    print(f'Parameters: {internal_params_labels + action_params_labels + sensory_params_labels}')

    # Import model dicts
    models_dict = import_states_params_asdict()


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
                                          selected_data,
                                          fitting_list,
                                          outfile_path=outfile)

    return summary





"""
High level function to fit parameters at the experiment level
Inputs:
    data_path: path to input data file (.obj containing a dictionary)
    experiments: experiments selected (as list of strings)
    internal_state: internal state name
    parameter_tags: parameter tags (as list of tags or string of tags, see below for possibilities)
    outdir_path: path to output data file (absolute path to target directory)

Output:
    pd.DataFrame: df containing fitting summary for the specified data, internal state and tags
"""

def fit_group_param_wrapper(data_path,
                            experiments,
                            internal_state,
                            parameter_tags,
                            outdir_path):
    
    exp_str = 'exp' + ''.join([experiment[-1] for experiment in experiments])
    
    # Define outfile path
    if outdir_path[-4:] == 'csv':
        outfile = outdir_path
    else:
        outfile = f'{outdir_path}/{exp_str}_{internal_state}_&_{"_".join(parameter_tags)}.csv'
    
    ## Import behavioural experiment
    with open(data_path, 'rb') as inFile:
        modelling_data = pickle.load(inFile)

    # Select participans
    selected_data = {}
    pick_interval = 1
    idx = 0
    for part, data in modelling_data.items():
        if data['experiment'] in experiments and idx % pick_interval == 0:
            selected_data[part] = data

        idx += 1

    # Pick states to fit
    internal_states_list = [internal_state]
    action_states_list = ['experience_vao']
    sensory_states_list = ['omniscient']

    fitting_change = False
    fitting_attention = False
    if 'change' in internal_state:
        if 'cha' in parameter_tags:
            fitting_change = True          
        
        if 'att' in parameter_tags:
            fitting_attention = True

    
    if 'LC_discrete' in internal_state:
        if 'att' in parameter_tags:
            fitting_attention = True  

    if 'Adaptive' in internal_state:
        if 'att' in parameter_tags:
            fitting_attention = True   

    # CES
    fitting_guess = False
    fitting_strength = False
    if internal_state[:3] == 'ces':
        if 'guess' in parameter_tags:
            fitting_guess = True 

        if 'str' in parameter_tags:
            fitting_strength = True

    # Prior
    fitting_prior = False
    if 'prior' in parameter_tags:
        fitting_prior = True
   
    random_increment = 1

    params_to_fit_tuple = params_to_fit_importer(internal_states_list[0], 
                                                 fitting_change=fitting_change,
                                                 fitting_attention=fitting_attention,
                                                 fitting_guess=fitting_guess,
                                                 fitting_strength=fitting_strength,
                                                 fitting_prior=fitting_prior,
                                                 random_increment=random_increment)
    params_initial_guesses = params_to_fit_tuple[0]
    params_bounds = params_to_fit_tuple[1]
    internal_params_labels = params_to_fit_tuple[2]
    action_params_labels = params_to_fit_tuple[3]
    sensory_params_labels = params_to_fit_tuple[4]
    fitting_list = params_to_fit_tuple[5]

    if len(experiments) == 1:
        exp = experiments[0]
    else:
        exp = experiments

    print(f'Fitting: {internal_states_list[0]} for {exp}...')
    print(f'Parameters: {internal_params_labels + action_params_labels + sensory_params_labels}')
    
    # Import model dicts
    models_dict = import_states_params_asdict()

    
    # Run fitting function
    summary = fit_params_models_grouplevel(params_initial_guesses,
                                           params_bounds,
                                           internal_params_labels,
                                           action_params_labels,
                                           sensory_params_labels,
                                           internal_states_list,
                                           action_states_list,
                                           sensory_states_list,
                                           models_dict,
                                           selected_data,
                                           fitting_list,
                                           exp,
                                           outfile_path=outfile)

    return summary