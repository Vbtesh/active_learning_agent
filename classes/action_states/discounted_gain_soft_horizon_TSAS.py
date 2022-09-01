from classes.action_states.action_state import Treesearch_AS
from copy import deepcopy

# Discounted gain soft horizon
class Discounted_gain_soft_horizon_TSAS(Treesearch_AS):
    def __init__(self, N, K, behaviour, epsilon, possible_actions, action_len, policy_funcs, C, knowledge, discount, horizon):

        self._discount = discount 
        self._horizon = horizon

        super().__init__(N, K, behaviour, epsilon, possible_actions, action_len, policy_funcs, C, knowledge, self._build_tree_dgsh, tree_search_func_args=[discount, horizon])


    def _build_tree_dgsh(self, gain, external_state, sensory_state, internal_state, gain_update_rule, discount, horizon, depth=0, seq=''):
        if depth > 0 and gain * discount**depth < horizon:
            return gain, seq
        else:
            new_tree = []
            for i in range(self._num_actions):
                new_gain, external_state_out, sensory_state_out, internal_state_out = gain_update_rule(i, 
                                                                                                       deepcopy(external_state),  
                                                                                                       deepcopy(sensory_state), 
                                                                                                       deepcopy(internal_state))
                acc_gain = gain + new_gain

                # Compile sequence
                if not seq:
                    new_seq = str(i)
                else:
                    new_seq = seq + ',' + str(i)

                new_leaf = self._build_tree_dgsh(acc_gain, external_state, sensory_state, internal_state, gain_update_rule, discount, horizon, depth+1, seq=new_seq)
                new_tree.append(new_leaf)

                # Roll back for next branch exploration
                #internal_state.rollback(self._action_len)
                #sensory_state.rollback(self._action_len)
                #external_state.reset(self._action_len)

            return new_tree