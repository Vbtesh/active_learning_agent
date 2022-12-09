from classes.action_states.action_state import Treesearch_AS
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