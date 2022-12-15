from classes.action_states.action_state import Treesearch_AS
from classes.action_states.action_state import Pseudo_AS
from copy import deepcopy

# Undiscounted gain hard horizon
class Variational_Actor_TSAS(Treesearch_AS):
    def __init__(self, N, K, behaviour, epsilon, possible_actions, action_len, policy_funcs, C, knowledge, gain_type, depth, resource_rational_parameter=0):
        self._depth = depth
        super().__init__(N, K, behaviour, epsilon, possible_actions, action_len, policy_funcs, C, knowledge, gain_type, self._build_tree_ughh, tree_search_func_args=[self._depth],  resource_rational_parameter=resource_rational_parameter)

    
    def _build_tree_ughh(self, gain, external_state, sensory_state, internal_state, gain_update_rule, depth, seq=''):
        if depth == 0:
            return gain, seq
        else:
            new_tree = []
            for i in range(self._num_actions):
                new_gain, external_state_out, sensory_state_out, internal_state_out = gain_update_rule(i, 
                                                                                                       deepcopy(external_state),  
                                                                                                       deepcopy(sensory_state), 
                                                                                                       deepcopy(internal_state))
                acc_gain = gain + new_gain

                a = self._remap_action(i)
                # Compile sequence
                if not seq:
                    new_seq = str(i)
                else:
                    new_seq = seq + ',' + str(i)

                new_leaf = self._build_tree_ughh(acc_gain, external_state_out, sensory_state_out, internal_state_out, gain_update_rule, depth-1, seq=new_seq)
                new_tree.append(new_leaf)

                ## Roll back for next branch exploration
                #internal_state.rollback(self._action_len + 1)
                #sensory_state.rollback(self._action_len + 1)
                #external_state.reset(self._action_len + 1)

            return new_tree

    def _run_local_experiment(self, action_idx, external_state, sensory_state, internal_state):
        # Same as the main function in the TreeSearch_AS object expect that only information gained about the factors of interest is considered.
        ## ACTUALLY UNNECESSARY? variational internal states only update so the only reduction in entropy is already controlled
        to_update = internal_state.factors_to_update
        init_entropy = internal_state.variational_posterior_entropy[to_update].sum()

        #print('init_entropy:', init_entropy, 'action:', action, 'external_state:', external_state.causal_vector)
        if internal_state._n + self._action_len + 1 >= internal_state._N:
            N = internal_state._N - internal_state._n
        else:
            N = self._action_len + 1

        action = self._remap_action(action_idx)

        # Instantiate pseudo action
        pseudo_action = Pseudo_AS(action, self._action_len, self._realised)

        if self._n > 1:
            a = action

        for n in range(int(N)):
            external_state.run(interventions=action)
            sensory_state.observe(external_state, internal_state)
            internal_state.update(sensory_state, pseudo_action)

        posterior_entropy = internal_state.variational_posterior_entropy[to_update].sum()

        gain = self._gain_function(action, init_entropy, posterior_entropy, sensory_state, internal_state)

        return (gain, external_state, sensory_state, internal_state)

