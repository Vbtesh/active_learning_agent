import numpy as np
from scipy.spatial.distance import pdist

def normalised_euclidean_distance(ground_truth, posterior):
    euc_dist = 1-pdist(np.stack((ground_truth, posterior)))[0] / np.linalg.norm(abs(np.array(ground_truth)) + 2*np.ones((1, 6)))
    return euc_dist


def read_array_str(array_str):
    return np.array([float(i) for i in array_str[1:-1].strip().split(' ') if len(i) > 0])


def distance(array1_str, array2_str):
    array1 = read_array_str(array1_str)
    array2 = read_array_str(array2_str)

    return normalised_euclidean_distance(array1, array2)


def concat_name(sign, trial_name):
    if sign in ['pos', 'neg', '1', '2', '3']:
        return trial_name + '_' + sign
    else:
        return trial_name
    

def transform_generic(graph_str, ground_truth, graph_name, transform_indirect):
    graph = read_array_str(graph_str)
    if graph_name.split('_')[0] in ['chain', 'dampened', 'confound']:
        #print(graph_name, read_array_str(ground_truth), '->', read_array_str(ground_truth) * transform_indirect[graph_name])
        return graph * transform_indirect[graph_name]
    else:
        return graph


def tag_column(array_str, tag_dict):
    return tag_dict[array_str]


def add_dampened_tag(df):
    confound_tags = {graph:str(i+1) for i, graph in enumerate(np.sort(df[df.trial_name == 'dampened'].ground_truth.unique()))}
    idx = df[df.trial_name == 'dampened'].index
    df_confound = df[df.trial_name == 'dampened']
    df.loc[idx, 'sign'] = df_confound.apply(lambda x: tag_column(x.ground_truth, confound_tags), axis=1).to_list()
    return df


def get_model_type(model, internal_state):
    if internal_state == 'mean_field_vis':
        model_type = 'Variational'
    else:
        model_type = 'Standard'
    return model_type

def get_model_factorisation(model, internal_state):
    if 'local_computations' in model or 'LC' in model:
        model_factorisation = 'LC'
    else:
        model_factorisation = 'Normative'

    return model_factorisation

def get_model_focus(model, internal_state):
    if 'att' in model or 'single_variable' in model or 'full' in model:
        model_interfocus = 'Interfocused'
    else:
        model_interfocus = 'Omniscient'

    return model_interfocus


def int_or_fl(value):
    try:
        value = float(value)
    except ValueError:
        try:
            value = int(value)
        except ValueError:
            pass
    return value


"""
Indirect effects
"""
## Indirect effects
def indirect_effects_vector(graph):
    out = np.array([graph[1]*graph[5], 
                    graph[0]*graph[3], 
                    graph[3]*graph[4], 
                    graph[2]*graph[1], 
                    graph[5]*graph[2], 
                    graph[4]*graph[0]])
    return out


def find_indirect_errors(gt, p):
    ie_gt = indirect_effects_vector(gt)
    gt_out = ie_gt + gt
    gt_out[np.abs(gt_out) > 1] = gt_out[np.abs(gt_out) > 1] / gt_out[np.abs(gt_out) > 1]

    return p[gt_out != gt] != gt[gt_out != gt], gt_out != gt, gt_out