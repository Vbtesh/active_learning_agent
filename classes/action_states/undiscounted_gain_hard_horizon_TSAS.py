from classes.action_states.action_state import Treesearch_AS

# Undiscounted gain hard horizon
class Undiscounted_gain_hard_horizon_TSAS(Treesearch_AS):
    def __init__(self, N, K, behaviour, possible_actions, action_len, policy_funcs, epsilon, C, knowledge, depth):
        self._depth = depth
        super().__init__(N, K, behaviour, possible_actions, action_len, policy_funcs, epsilon, C, knowledge, self._build_tree_ughh, tree_search_func_args=[self._depth])

    
    def _build_tree_ughh(self, gain, external_state, sensory_state, internal_state, gain_update_rule, depth, seq=''):
        if depth == 0:
            return gain, seq
        else:
            new_tree = []
            for i in range(self._num_actions):
                new_gain = gain_update_rule(i, external_state, sensory_state, internal_state)
                acc_gain = gain + new_gain

                # Compile sequence
                if not seq:
                    new_seq = str(i)
                else:
                    new_seq = seq + ',' + str(i)

                new_leaf = self._build_tree_ughh(acc_gain, external_state, sensory_state, internal_state, gain_update_rule, depth-1, seq=new_seq)
                new_tree.append(new_leaf)

                # Roll back for next branch exploration
                internal_state.rollback(self._action_len)
                sensory_state.rollback(self._action_len)
                external_state.reset(self._action_len)

            return new_tree