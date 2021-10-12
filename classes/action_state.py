import numpy as np
import jax
from numpy.random.mtrand import uniform


class Action_state():
    def __init__(self, N, K, 
                          behaviour,
                          possible_actions, 
                          idle,
                          action_len, 
                          policy,
                          pmf_policy, 
                          epsilon, 
                          action_value_func):

        self._N = N
        self._n = 0
        self._K = K
        self._behaviour = behaviour

        if idle:
            self._num_actions = self._K * len(possible_actions) + 1
        else:
            self._num_actions = self._K * len(possible_actions)

        self._poss_actions = possible_actions
        self._action_len = action_len
        self._action_idx = 0
        self._current_action = None

        # Certainty parameter, stops intervening when posterior entropy is below the threshold
        self._epsilon = epsilon
 
        # A function that takes action values as arguments and return an sequence of actions which is then remapped onto action_seqs and then split using the remap_action function
        # Argument must be action values over all possible actions
        self._policy = policy 
        # Return the probability of taking an action given the current action values
        self._pmf_policy = pmf_policy

        # Action value function
        # Arguments must be: action_state or self, external_state, sensory_state, internal_state
        self._compute_action_values = action_value_func

        # Action values history
        self._action_values = [None for i in range(self._N+1)]
        self._action_seqs = [None for i in range(self._N+1)]

        self._planned_actions = [None for i in range(self._N+1)]


    # Core method, samples an action by computing action values and selecting one action according to the given policy
    def sample(self, external_state, sensory_state, internal_state):
        # If behaviour observer, return None, else if behaviour is random, return a random action
        if self._behaviour == 'obs':
            self._n += 1
            return None
        elif internal_state.posterior_entropy < self._epsilon and self._n > 0.33*self._N:
            self._n += 1
            return None

        self._action_idx += 1
        if self._action_idx >= self._action_len or self._n == 0:
            # If policy is random return random action
            if self._behaviour == 'random':
                self._current_action = self._remap_action(np.random.choice(self._num_actions))
            else:
                # Do simulation to estimate action values if policy is not random
                self._action_values[self._n], action_seqs = self._compute_action_values(external_state, sensory_state, internal_state)
                # Sample a sequence of actions
                sampled_sequence_idx = self._policy(self._action_values[self._n])

                # Sample an sequence of actions from a policy function
                action_sequence = action_seqs[sampled_sequence_idx]

                print('argmax:', np.argmax(self._action_values[self._n]), 'argmax seq:', action_seqs[np.argmax(self._action_values[self._n])], 'sampled action:', sampled_sequence_idx, 'sampled sequence:', action_sequence)

                action_seq_int = [int(a) for a in action_sequence.split(',')]
                self._planned_actions[self._n] = action_seq_int
                self._current_action = self._remap_action(action_seq_int[0])             

            # Reset action idx
            self._action_idx = 0

        self._n += 1
        return self._current_action


    # Rollback action state
    ## Used mostly for action selection
    def rollback(self, back=np.Inf):
        if back > self._N or back > self._n:
            self._n = 0

            # Reset Action values history, Action seq hist and planned actions
            self._action_values = [None for i in range(self._N+1)]
            self._action_seqs = [None for i in range(self._N+1)]
            self._planned_actions = [None for i in range(self._N+1)]
        else:
            self._n -= back

            # Reset Action values, seq and planned action from n to N
            for n in range(self._n+1, self._N+1):
                self._action_values[n] = None
                self._action_seqs[n] = None
                self._planned_actions[n] = None

    
    # Return None, for idleness or a tuple (variable index, variable value) otherwise
    def _remap_action(self, action):
        if action // self._poss_actions.size > self._K - 1:
            return None
        else:
            variable_idx = action // self._poss_actions.size
            variable_value = self._poss_actions[action % self._poss_actions.size]

            return (variable_idx, variable_value)



# Tree search action selection
class Treesearch_AS(Action_state):
    def __init__(self, N, K, behaviour, possible_actions, idle, action_len, policy, pmf_policy, epsilon, C, knowledge, tree_search_func, tree_search_func_args=[]):
        super().__init__(N, K, behaviour, possible_actions, idle, action_len, policy, pmf_policy, epsilon, self._tree_search_action_values)

        self._tree_search_func = tree_search_func
        self._tree_search_func_args = tree_search_func_args
        
        self._knowledge = knowledge
        self._C = C        
        

    def _tree_search_action_values(self, external_state, sensory_state, internal_state):

        # Logic for tree search based action values
        true_graph = external_state.causal_matrix


        for c in range(self._C):

            # Sample graph from posterior or use knowledge
            if type(self._knowledge) == np.ndarray:
                # Use internal_state passed as knowledge argument
                external_state.causal_matrix = internal_state._causality_matrix(self._knowledge, fill_diag=1)
            elif self._knowledge == 'perfect':
                external_state.causal_matrix = true_graph
            elif self._knowledge == 'random':
                # Sample a internal_state from a uniform distribution
                graph_c = internal_state.posterior_sample(uniform=True, as_matrix=True)
                external_state.causal_matrix = graph_c
            else:
                # Sample a internal_state from the current posterior
                graph_c = internal_state.posterior_sample(as_matrix=True)
                external_state.causal_matrix = graph_c

            # Variable for printing, sample has 
            sample_print = external_state.causal_vector
            print('Compute action values, C=', c, 'Model n:', internal_state._n, 'Sampled graph:', sample_print)

            # Build outcome tree
            action_values_astree = self._tree_search_func(0, external_state, 
                                                             sensory_state,
                                                             internal_state,
                                                             self._run_local_experiment,
                                                             *self._tree_search_func_args)

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


    # Background methods
    # Update rule for the leaves values
    # Simulates an agent's update
    def _run_local_experiment(self, action_idx, external_state, sensory_state, internal_state):
        init_entropy = internal_state.posterior_entropy
        #print('init_entropy:', init_entropy, 'action:', action, 'external_state:', external_state.causal_vector)
        if internal_state._n + self._action_len >= internal_state._N:
            N = internal_state._N - internal_state._n
        else:
            N = self._action_len

        action = self._remap_action(action_idx)

        for n in range(N):
            x = external_state.run(interventions=action)
            sensory_state.observe(external_state, internal_state)
            internal_state.update(sensory_state, action)

        return init_entropy - internal_state.posterior_entropy



# Undiscounted gain hard horizon
class Undiscounted_gain_hard_horizon_TSAS(Treesearch_AS):
    def __init__(self, N, K, behaviour, possible_actions, idle, action_len, policy, pmf_policy, epsilon, C, knowledge, depth):
        self._depth = depth
        super().__init__(N, K, behaviour, possible_actions, idle, action_len, policy, pmf_policy, epsilon, C, knowledge, self._build_tree_ughh, tree_search_func_args=[self._depth])

    
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


# Discounted gain soft horizon
class Discounted_gain_soft_horizon_TSAS(Treesearch_AS):
    def __init__(self, N, K, behaviour, possible_actions, idle, action_len, policy, pmf_policy, epsilon, C, knowledge, discount, horizon):

        self._discount = discount 
        self._horizon = horizon

        super().__init__(N, K, behaviour, possible_actions, idle, action_len, policy, pmf_policy, epsilon, C, knowledge, self._build_tree_dgsh, tree_search_func_args=[discount, horizon])

    
    def _build_tree_dgsh(self, gain, external_state, sensory_state, internal_state, gain_update_rule, discount, horizon, depth=0, seq=''):
        if depth > 0 and gain * discount**depth < horizon:
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

                new_leaf = self._build_tree_dgsh(acc_gain, external_state, sensory_state, internal_state, gain_update_rule, discount, horizon, depth+1, seq=new_seq)
                new_tree.append(new_leaf)

                # Roll back for next branch exploration
                internal_state.rollback(self._action_len)
                sensory_state.rollback(self._action_len)
                external_state.reset(self._action_len)

            return new_tree
