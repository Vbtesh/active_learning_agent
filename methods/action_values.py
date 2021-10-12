import jax
import numpy as np

# Update rule for the leaves values
def run_local_experiment(action_idx, external_state, sensory_state, internal_state, action_state):
    init_entropy = internal_state.posterior_entropy
    #print('init_entropy:', init_entropy, 'action:', action, 'external_state:', external_state.causal_vector)
    x_prev = external_state.data_last
    if internal_state._n + action_state._action_length >= internal_state._N:
        N = internal_state._N - internal_state._n

    action = action_state.remap_action(action_idx)

    for n in range(N):
        x = external_state.run(interventions=action)
        internal_state.update(x_prev, x, action)
        x_prev = x

    return init_entropy - internal_state.posterior_entropy


# General wrapper for tree search based action values
def generate_tree_search_action_values(tree_search_func, tree_search_func_args, C, knowledge, gain_update_rule=run_local_experiment):

    def tree_search_action_values(external_state, 
                                  sensory_state, 
                                  internal_state,
                                  action_state):

        # Logic for tree search based action values
        posterior_graphs = internal_state.posterior_models
        true_graph = external_state.causal_matrix


        for c in range(C):

            # Sample graph from posterior or use knowledge
            if type(knowledge) == np.ndarray:
                # Use internal_state passed as knowledge argument
                external_state.causal_matrix = internal_state._causality_matrix(knowledge, fill_diag=0)
            elif knowledge == 'random':
                # Sample a internal_state from a uniform distribution
                g_c = np.random.choice(np.arange(posterior_graphs.size))
                external_state.causal_matrix = internal_state._sample_space_as_mat[g_c,:,:] 
            elif knowledge == 'perfect':
                external_state.causal_matrix = true_graph
            else:
                # Sample a internal_state from the current posterior
                g_c = np.random.choice(np.arange(posterior_graphs.size), p=posterior_graphs)
                external_state.causal_matrix = internal_state._sample_space_as_mat[g_c,:,:]

            # Variable for printing, sample has 
            sample_print = external_state.causal_vector
            print('Compute action values, C=', c, 'Model n:', internal_state._n, 'Sampled graph:', sample_print)

            # Build outcome tree
            action_values_astree = tree_search_func(0, external_state, 
                                                       sensory_state,
                                                       internal_state, 
                                                       action_state,
                                                       gain_update_rule,
                                                       *tree_search_func_args)

            # Extract action values
            leaves = jax.tree_leaves(action_values_astree)
            leaves_table = np.array(leaves).reshape((int(len(leaves)/2), 2))
            action_values_c, seqs = leaves_table[:, 0].astype(float), leaves_table[:, 1]

            # Update action_value for time n
            if c == 0:
                action_values = action_values_c
                action_seqs = seqs
            else:
                action_values += 1/(c+1) * (action_values_c - action_values)

        return action_values, action_seqs

    # Return generated function
    return tree_search_action_values


# Tree search functions

# Undiscounted gain hard horizon
def build_tree_ughh(gain, external_state, sensory_state, internal_state, action_state, gain_update_rule, depth, seq=''):
    if depth == 0:
        return gain, seq
    else:
        new_tree = []
        for i in range(action_state._num_actions):
            new_gain = gain_update_rule(i, external_state, sensory_state, internal_state, action_state)
            acc_gain = gain + new_gain

            # Compile sequence
            if not seq:
                new_seq = str(i)
            else:
                new_seq = seq + ',' + str(i)
    
            new_leaf = build_tree_ughh(acc_gain, external_state, sensory_state, internal_state, action_state, gain_update_rule, depth-1, seq=new_seq)
            new_tree.append(new_leaf)

            # Roll back for next branch exploration
            internal_state.rollback(action_state._action_len)
            external_state.reset(action_state._action_len)

        return new_tree


# Discounted gains soft horizon
def build_tree_dgsh(gain, external_state, sensory_state, internal_state, action_state, gain_update_rule, discount, horizon, depth, seq=''):
    if depth > 0 and gain * discount**depth < horizon:
        return gain, seq
    else:
        new_tree = []
        for i in range(action_state._num_actions):
            new_gain = gain_update_rule(i, external_state, sensory_state, internal_state, action_state)
            acc_gain = gain + new_gain

            # Compile sequence
            if not seq:
                new_seq = str(i)
            else:
                new_seq = seq + ',' + str(i)
    
            new_leaf = build_tree_dgsh(acc_gain, external_state, sensory_state, internal_state, action_state, gain_update_rule, discount, horizon, depth-1, seq=new_seq)
            new_tree.append(new_leaf)

            # Roll back for next branch exploration
            internal_state.rollback(action_state._action_len)
            external_state.reset(action_state._action_len)

        return new_tree



## NON CRITICAL TREE SEARCH FUNCTION
def build_tree_generic(gain, leaves, depth, update_rule, *args):
    if depth == 0:
        return gain
    else:
        new_tree = []
        for i in range(leaves):
            new_gain = update_rule(i, *args)
            acc_gain = gain + new_gain
            new_tree.append(build_tree_generic(acc_gain, leaves, depth-1, update_rule, *args))
        return new_tree


def build_tree_seq(leaves, depth, seq=''):
    if depth == 0:
        return seq
    else:
        new_tree = []
        for i in range(leaves):
            if not seq:
                new_seq = str(i)
            else:
                new_seq = seq + ',' + str(i)
            new_tree.append(build_tree_seq(leaves, depth-1, seq=new_seq))
        return new_tree


def build_tree_combined(gain, leaves, update_rule, discount, horizon, depth, seq=''):
    if depth > 0 and gain * discount**depth < horizon:
        return gain, seq
    else:
        new_tree = []
        for i in range(leaves):
            new_gain = update_rule(i)
            acc_gain = gain + new_gain

            if not seq:
                new_seq = str(i)
            else:
                new_seq = seq + ',' + str(i)

            new_tree.append(build_tree_combined(acc_gain, leaves, update_rule, discount, horizon, depth+1, seq=new_seq))
        return new_tree
