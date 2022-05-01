import numpy as np
import jax
from copy import deepcopy
from numpy.random.mtrand import uniform


class Action_state():
    def __init__(self, N, K, 
                          behaviour,
                          epsilon, 
                          sample_action_func,
                          fit_action_func):

        self._N = N
        self._n = 0
        self._K = K
        self._behaviour = behaviour

        # Sets up the length and associated attribute 
        self._action_len = None
        
        # Default current action
        self._current_action = None

        # Certainty parameter, stops intervening when posterior entropy is below the threshold
        self._epsilon = epsilon

        # Action value function
        # Arguments must be: action_state or self, external_state, sensory_state, internal_state
        # Returns a tuple representing an action or None for idleness
        self._sample_action = sample_action_func
        # Argument are action as tuple or None, external_state, sensory_state, internal_state
        # Returns a log probability of taking the action
        self._fit_action = fit_action_func
        # Action len fit
        self._action_len_fit = None
        self._len_fit = False
        self._action_len_real = None
        self._len_real = False

        # Action history
        ## Generic list for any action specification (depends on model)
        self._actions_history = [None for i in range(self._N+1)]
        ## Variables intervened on for future fitting
        self._variable_history = np.zeros(self._N+1)
        self._variable_history[:] = np.nan
 
        # Set realised to False by default
        self._realised = False


    # Core method, samples an action by computing action values and selecting one action according to the given policy
    def sample(self, external_state, sensory_state, internal_state):
        # If behaviour observer, return None, else if behaviour is random, return a random action
        if self._behaviour == 'obs' and not self._realised:
            self._n += 1
            return None
        elif self._behaviour == 'obs' and self._realised:
            self._current_action = self.a_real
            if type(self._current_action) == tuple:
                self._variable_history[self._n] = self._current_action[0]

            self._n += 1
            return self._current_action
        elif self._behaviour == 'random':
            variable = np.random.choice([i for i in range(self._K)] + [None])
            if not variable:
                self._current_action = variable
            else:
                value = np.random.choice([-1, 1]) * np.random.choice(self._poss_actions)
                self._current_action = [variable, value]

            if type(self._current_action) == tuple:
                self._variable_history[self._n] = self._current_action[0]

            self._n += 1
            return self._current_action

        elif internal_state.posterior_entropy < self._epsilon and self._n > 0.33*self._N:
            self._n += 1
            return None

        # Sample_action function.
        self._current_action = self._sample_action(external_state, sensory_state, internal_state)

        if type(self._current_action) == tuple:
            self._variable_history[self._n] = self._current_action[0]

        self._n += 1
        return self._current_action
    

    # Fit action to action states
    ## Needs action data to be loaded to function
    def fit(self, external_state, sensory_state, internal_state): 
        # Constraint actual action
        constrained_action = self._constrain_action(self.a_fit)
        self._current_action = constrained_action

        # Get action len from a_fit by looking ahead in the array
        # Save it for fitting certain internal states
        if self.a_real and not self._len_real:
            self._action_len_real = self._get_action_len(self.a_real)
            self._len_real = True
        elif not self.a_real and self._len_real:
            self._len_real = False

          
        if self._behaviour == 'obs':
            # Record action if action was taken
            self._actions_history[self._n] = self._current_action 
            if type(self._current_action) == tuple:
                self._variable_history[self._n] = self._current_action[0]
            
            # Do not fit actions
            self._n += 1
            self._log_likelihood += 0
            self._log_likelihood_history[self._n] = self._log_likelihood
            return 0  # Log probability of acting given that the person is an observer is necessarily - infinity

        # If action and action fit are different, do not penalise log likelihood
        if self.a_real and not self.a_fit:
            self._n += 1
            self._log_likelihood += 0
            self._log_likelihood_history[self._n] = self._log_likelihood
            return 0

        if self._behaviour == 'random':
            # If behaviour is random, simply return the probability of taking any action
            action_log_prob = np.log(1 / self._num_actions)

            self._log_likelihood += action_log_prob

            # Update history
            self._actions_history[self._n] = self._current_action 
            if type(self._current_action) == tuple:
                self._variable_history[self._n] = self._current_action[0]
            self._log_likelihood_history[self._n] = self._log_likelihood

            self._n += 1
            return action_log_prob
        else:
            # Compute and return Log probability of action
            action_log_prob = self._fit_action(self._current_action, external_state, sensory_state, internal_state)
            # Update log likelihood
            self._log_likelihood += action_log_prob

            # Update hitory
            self._actions_history[self._n] = self._current_action 
            if type(self._current_action) == tuple:
                self._variable_history[self._n] = self._current_action[0]
            self._log_likelihood_history[self._n] = self._log_likelihood

            self._n += 1
            return action_log_prob


    # Load data
    def load_action_data(self, actions, actions_fit, variables_values):
        self._A = actions
        self._A_fit = actions_fit

        # Create lists of actions indices for each variables
        self._A_fit_indices = self._split_action_array(self._A_fit)
        self._A_real_indices = self._split_action_array(self._A)
        
        self._X = variables_values

        self._log_likelihood = 0
        self._log_likelihood_history = np.zeros(self._N + 1)

        self._realised = True

    # Load data
    ## Only works with run
    def load_action_plan(self, actions, variables_values):
        self._A = actions

        self._A_real_indices = self._split_action_array(self._A)
        
        self._X = variables_values

        self._realised = True
        
    # Rollback action state
    ## Used mostly for action selection
    def rollback(self, back=np.Inf):
        if back > self._N or back > self._n:
            self._n = 0

            # Reset Action values history, Action seq hist and planned actions
            self._action_values = [None for i in range(self._N+1)]
            self._actions_history = [None for i in range(self._N+1)]
        else:
            self._n -= back

            # Reset Action values, seq and planned action from n to N
            for n in range(self._n+1, self._N+1):
                self._action_values[n] = None
                self._actions_history[n] = None


    @property
    def a(self):
        return self._current_action

    @property
    def a_real(self):
        if np.isnan(self._A[self._n]):
            return None
        else: 
            action = int(self._A[self._n])
            return (action, self._X[self._n,:][action])

    @property
    def a_fit(self):
        if np.isnan(self._A_fit[self._n]):
            return None
        else:
            action = int(self._A_fit[self._n])
            return (action, self._X[self._n,:][action])
    
    @property
    def a_len(self):
        return self._action_len

    @property
    def a_len_fit(self):
        return self._get_action_len(self.a_fit)
    
    @property
    def a_len_real(self):
        return self._get_action_len(self.a_real)

    @property
    def actions(self):
        return self._A[0:self._n+1]

    @property
    def actions_fit(self):
        return self._A_fit[0:self._n+1]
    
    @property
    def realised(self):
        return self._realised

    # Evaluate action similarity
    def _action_check(self, action_1, action_2):
        if self._constrain_action(action_2) == self._constrain_action(action_1):
            return True
        else:
            return False

    def _constrain_action(self, action):
        if not action:
            return None
        else:
            set_value_idx = np.argmin(np.abs(self._poss_actions - action[1]))
            return (action[0], self._poss_actions[set_value_idx])

    # Methods for properly fitting change based models
    def _split_action_array(self, action_array):
        action_lists = []
        for k in range(self._K):
            action_k_indices = np.where(action_array == k)[0]
            action_lists.append(self._consecutive(action_k_indices))
        return action_lists

    # Get length of current action
    def _get_action_len(self, action, which='real'):
        if which == 'real':
            for a in self._A_real_indices[action[0]]:
                if self._n in a:
                    return a.size
        else:
            for a in self._A_fit_indices[action[0]]:
                if self._n in a:
                    return a.size

    # Split array of indices into seperate arrays of consecutives indices (consecutive is specific by the stepsize parameter)
    def _consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
       


# Tree search action selection
class Treesearch_AS(Action_state):
    def __init__(self, N, K, behaviour, epsilon, possible_actions, action_len, policy_funcs, C, knowledge, tree_search_func, tree_search_func_args=[]):
        super().__init__(N, K, behaviour, epsilon, self._tree_search_action_sample, self._tree_search_action_fit)

        # Num of possible action is all possible values for all variable plus 1 for staying idle
        self._num_actions = self._K * len(possible_actions) + 1

        self._action_grid = np.arange(self._K * len(possible_actions)).reshape(self._K, len(possible_actions))

        self._poss_actions = possible_actions
        self._action_idx = 0
        self._current_action = self._num_actions # Represents the index of doing nothing, i.e. None, in list form 

        self._tree_search_func = tree_search_func
        self._tree_search_func_args = tree_search_func_args

        # Action sampling parameters
        self._action_idx = 0
        self._action_len = action_len

        # A function that takes action values as arguments and return an sequence of actions which is then remapped onto action_seqs and then split using the remap_action function
        # Argument must be action values over all possible actions
        self._policy = policy_funcs[0]
        # Return the probability of taking an action given the current action values
        self._pmf_policy = policy_funcs[1]
        # Return the parameters of the policy given the the current action values
        self._params_policy = policy_funcs[2]

        # Sets up specific histories
        self._action_values = [None for i in range(self._N+1)]
        self._action_seqs_values = [None for i in range(self._N+1)]
        self._action_seqs = [None for i in range(self._N+1)]
        
        self._knowledge = knowledge
        self._C = C    


    def _tree_search_action_sample(self, external_state, sensory_state, internal_state):  
        # If time to act, sample new action
        if self._action_idx >= self._action_len or self._n == 0:
            # Reset action_idx
            self._action_idx = 0
            
            # If behaviour is random, return a random action
            if self._behaviour == 'random':
                return self._remap_action(np.random.choice(self._num_actions))

            # Compute action values
            ## /!\ Deepcopies are important to not break the main state objects /!\
            action_values = self._tree_search_action_values(deepcopy(external_state), 
                                                            deepcopy(sensory_state), 
                                                            deepcopy(internal_state))

            # Sample a sequence of actions
            sampled_action = self._policy(action_values)

            # Update history
            self._action_values[self._n] = action_values  
            self._actions_history[self._n] = sampled_action

            # Reset action_idx
            self._action_idx = 0

            # Return action remapped to the action (idx) to tuple (variable, value), action len, obs len
            return self._remap_action(sampled_action)
       
        else:
            self._action_idx += 1
            # If not time to act again, simply return None
            return None


    def _tree_search_action_fit(self, action, external_state, sensory_state, internal_state):   
        # Flatten action
        flat_action = self._flatten_action(action)

        # Compute action values
        ## /!\ Deepcopies are important to not break the main state objects /!\
        action_values = self._tree_search_action_values(deepcopy(external_state), 
                                                        deepcopy(sensory_state), 
                                                        deepcopy(internal_state))

        # Compute policy params
        action_prob = self._pmf_policy(flat_action, action_values)

        # Log of probability of action
        action_log_prob = np.log(action_prob)

        # Update history
        self._action_values[self._n] = action_values 
        self._action_len_history[self._n] = self._action_len  

        # Return action remapped to the action (idx) to tuple (variable, value)
        return action_log_prob
        

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
            seqs_values_astree = self._tree_search_func(0, external_state, 
                                                             sensory_state,
                                                             internal_state,
                                                             self._run_local_experiment,
                                                             *self._tree_search_func_args)

            # Extract action values
            leaves = jax.tree_leaves(seqs_values_astree)
            leaves_table = np.array(leaves).reshape((int(len(leaves)/2), 2))
            seqs_values_c, seqs = leaves_table[:, 0].astype(float), leaves_table[:, 1]

            # Update action_value for time n
            if c == 0:
                seqs_values = seqs_values_c
                action_seqs = seqs
            else:
                seqs_values += 1/(c+1) * (seqs_values_c - seqs_values)


        self._action_seqs_values[self._n] = seqs_values
        self._action_seqs[self._n] = action_seqs  

        # Average over values
        action_values = self._average_over_sequences(seqs_values, seqs)

        return action_values


    # Background methods
    # Return None, for idleness or a tuple (variable index, variable value) otherwise
    def _remap_action(self, action):
        if action // self._poss_actions.size > self._K - 1:
            return None
        else:
            variable_idx = action // self._poss_actions.size
            variable_value = self._poss_actions[action % self._poss_actions.size]

            return (variable_idx, variable_value)

    def _flatten_action(self, action):
        if not action:
            return self._num_actions - 1
        else:    
            value_idx = np.argmax(np.where(self._poss_actions == action[1])[0])
            return self._action_grid[action[0], value_idx]
    
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
            external_state.run(interventions=action)
            sensory_state.observe(external_state, internal_state)
            internal_state.update(sensory_state, action)

        return init_entropy - internal_state.posterior_entropy

    
    def _average_over_sequences(self, seqs_values, seqs):
        first_action_in_seq = np.array([int(seq.split(',')[0]) for seq in seqs])
        action_values = np.zeros(self._num_actions)
        for i in range(self._num_actions):
            action_values[i] = seqs_values[first_action_in_seq == i].sum()

        return action_values


# Experience based action selection
## Action values are considered state independent
class Experience_AS(Action_state):
    def __init__(self, N, K, behaviour, epsilon, possible_actions, policy_funcs, experience_gained_func):
        super().__init__(N, K, behaviour, epsilon, self._experience_action_sample, self._experience_action_fit)

        # Possible values for actions
        self._poss_actions = possible_actions

        # A function that takes action values as arguments and return an sequence of actions which is then remapped onto action_seqs and then split using the remap_action function
        # Argument must be action values over all possible actions
        self._policy = policy_funcs[0]
        # Return the probability of taking an action given the current action values
        self._pmf_policy = policy_funcs[1]
        # Return the parameters of the policy given the the current action values
        self._params_policy = policy_funcs[2]

        # Specific experience gain function
        self._experience_gained_func = experience_gained_func

        # Set up action values
        self._action_values = [None for i in range(self._N+1)]
        self._prob_variable = np.zeros((self._N+1, self._K))
        self._prob_sign = np.zeros((self._N+1, 2))

        # Acting variables
        self._current_action = None
        self._action_len = 0
        self._obs_len = 0

        # Variables for action tracking
        self._action_idx = 0
        self._obs_idx = 0
        self._acting = False
        self._last_planned_action = None

        # Variables for action fitting
        self._mean_value_taken = None
        self._previous_action = None


    # Action sampling function
    ## Return None if no obs or no action, action otherwise
    ## Does not depend on external states
    def _experience_action_sample(self, external_state, sensory_state, internal_state):
        if self._acting:
            
            # If not done acting, do sign change check and return
            if self._action_idx < self._action_len:

                # Sample sign if at least 1 sec has elapsed
                if self._action_idx % (1/self._time_unit) == 0:
                    if self._behaviour == 'random':
                        sign = np.random.choice([-1, 1])
                    else:
                        self._prob_sign[self._n, :] = self._compute_sign_change_prob(sensory_state)
                        sign = np.random.choice([-1, 1], p=self._prob_sign[self._n, :])

                    sampled_action = (self._current_action[0], sign*self._current_action[1])
                else:
                    sampled_action = self._current_action

                # Update history
                self._actions_history[self._n] = [sampled_action, self._action_len, self._obs_len]
                
                # Increment action index
                self._action_idx += 1

                return sampled_action

            # If done acting, do nothing
            ## Reset acting variables
            self._acting = False
            self._action_idx = 0

            ## Increment observing index
            self._obs_idx += 1

            return None
        
        else:
            # If still observing, do nothing
            if self._obs_idx < self._obs_len:
                # Increment obs index
                self._obs_idx += 1
                
                return None
            
            # If done observing, sample new action
            # If behaviour is random, sample at random:
            if self._behaviour == 'random':
                ## Sample variable
                variable = np.random.choice(np.arange(self._K))
                ## Sample action
                sampled_value = np.random.choice(self._poss_actions)
                sampled_length = np.random.choice(np.arange(self._max_acting_time))
                sampled_obs = np.random.choice(np.arange(self._max_obs_time))
                ## Sample sign
                sign = np.random.choice([-1, 1])

                # Combine action
                sampled_action = (variable, sign*sampled_value)             
            
            # Else, sample properly action according to model
            else:
                # Compute action values
                action_values = self._experience_gained_func(self._last_planned_action, self._action_len, self._obs_len, sensory_state, internal_state) 

                # Sample an action
                ## Sample variable
                p_var = self._compute_variable_prob(internal_state)
                variable = np.random.choice(np.arange(p_var.size), p=p_var)
                ## Sample action
                sampled_args = self._policy(action_values)
                sampled_action_idx, sampled_length, sampled_obs = sampled_args[0][0], sampled_args[1][0], sampled_args[2][0]

                ## Sample sign
                self._prob_sign[self._n, :] = self._compute_sign_change_prob(sensory_state)
                sign = np.random.choice([-1, 1], p=self._prob_sign[self._n, :])

                # Combine action
                sampled_action = (variable, sign*self._poss_actions[sampled_action_idx])

                # Update action value history and local params
                self._action_values[self._n] = action_values  


            # Convert length and obs to datapoint
            self._action_len = sampled_length / self._time_unit
            self._obs_len = sampled_obs / self._time_unit

            # Update action len and obs history
            self._last_planned_action = sampled_action
            self._actions_history[self._n] = [sampled_action, self._action_len, self._obs_len]

            # Reset observing index and toggle acting if length > 0
            self._obs_idx = 0
            if self._action_len > 0:
                self._acting = True
                self._action_idx += 1
            
            # Return action remapped to the action (idx) to tuple (variable, value)
            return sampled_action


    # Logic is complex as the likelihood of performing an action has to be computed for the index - action length
    # Alternative fitting function:
    # If action, compute prob of value and sign. else return 0.
    # When intervention is done, compute prob variable and prob action and obs len

    # action values update are always with respect to the last action taken, i.e. will always be based on, discounted by the number of step 
    # If no action taken yet, return 0
    # If action taken, but not acting, compute action values, return 0
    # If new action, acting or non acting, return p(new action)
    # If acting, no action, stop acting, compute action values, return 0


    def _experience_action_fit(self, action, external_state, sensory_state, internal_state):
        if not self._previous_action and not self._current_action:
            # Agent is idle and has not taken any action
            return 0
        elif self._previous_action and not self._current_action:
            # Start of observation time following action time, no acting done here
            self._acting = False
            # Increment observed length
            self._obs_len += 1
            return 0
        elif not self._previous_action and self._current_action:
            # First action is being taken, do nothing until action is completed
            self._acting = True
            
            # Set up book keeping for values taken and previous action.
            self._mean_value_taken = np.abs(self._current_action[1])
            self._previous_action = self._current_action
            # Set action length to 1
            self._action_len = 1
            return 0
        else:
            abs_current_action = (np.abs(self._current_action[0]), self._current_action[1])
            abs_prev_action = (np.abs(self._previous_action[0]), self._previous_action[1])
            # Cases where previous and current action are true
            ## If actions are the same, differentiate between pursuing the same action and reselecting it
            # Two cases
            # If new intervention: value previous action
            # Else if same intervention: compute sign prob if action_len % 5 == 0 else return 0

            # If last current action is not an action then, we are in a new intervention, otherwise, it is the same intervention
            if not self._actions_history[self._n-1]:
                # Get effective previous action: tuple with variable index and absolute mean value taken
                effective_previous_action = self._constrain_action((self._previous_action[0], self._mean_value_taken))
                # Changing action, compute action values for previous action
                action_values = self._experience_gained_func(effective_previous_action, self._action_len, self._obs_len, sensory_state, internal_state)
                self._action_values[self._n] = action_values 

                # Fit previous action 
                action_idx = self._flatten_action(effective_previous_action)
                action_len_sec = int(self._action_len * self._time_unit) # Round to nearest unit
                obs_len_sec = int(self._obs_len * self._time_unit) # Round to nearest unit

                # Get action probability from policy
                action_taken = (action_idx, action_len_sec, obs_len_sec)
                action_prob = self._pmf_policy(action_taken, self._action_values[self._n])

                # Previous action becomes current action, reset counters of length
                # We start new action here
                self._previous_action = self._current_action
                self._mean_value_taken = np.abs(self._current_action[1])
                self._action_len = 1
                self._obs_len = 0
                
                # Fit variable intervened upon, function of link entropy, higher entropy => higher prob
                # PROBLEM, THIS NEEDS TO BE FIT FOR CURRENT ACTION -- OK
                self._prob_variable[self._n, :] = self._compute_variable_prob(internal_state)
                var_prob = self._prob_variable[self._n, :][self._previous_action[0]]

                # Fit sign of intervention value
                # CAREFUL, THIS CAN CHANGE THROUGHOUT THE INTERVENTION AND SHOULD BE COMPUTED FOR EACH SECOND
                # NOT FOR PREVIOUS ACTION BUT CURRENT ACTION -- OK
                self._prob_sign[self._n, :] = self._compute_sign_change_prob(sensory_state)
                sign_prob = self._prob_sign[self._n, 0] if self._previous_action[1] > 0 else self._prob_sign[self._n, 1]
        
                # Returned are the log likelihood of taking previous action (value, action len, obs len)
                # But the log likelihood of picking the new variable and the observed sign
                full_action_log_prob = np.log(action_prob) + np.log(var_prob) + np.log(sign_prob)
                return full_action_log_prob[0]
            
            # Else if still acting and in the same intervention
            ## 2 cases:
            ### 1. Same value: 

            elif self._acting and self._action_len % (1 / self._time_unit) == 0:
                # If acting and 1 second has passed, compute current action sign prob and return log of prob
                self._prob_sign[self._n, :] = self._compute_sign_change_prob(sensory_state)
                sign_prob = self._prob_sign[self._n, 0] if self._current_action[1] > 0 else self._prob_sign[self._n, 1]

                # Increment action len, update mean value
                self._action_len += 1
                # The update is simply an arithmetic mean of all values taken expressed as an update and where the learning rate is the action length
                self._mean_value_taken = self._mean_value_taken + 1/self._action_len * (np.abs(self._current_action[1]) - self._mean_value_taken)
                return np.log(sign_prob)
                
            else:
                # If still intervening but not changing sign, increment action len, update mean value and return 0
                self._action_len += 1
                # The update is simply an arithmetic mean of all values taken expressed as an update and where the learning rate is the action length
                self._mean_value_taken = self._mean_value_taken + 1/self._action_len * (np.abs(self._current_action[1]) - self._mean_value_taken)
                return 0

                
                


    def _compute_variable_prob(self, internal_state):
        entropy_over_links = internal_state.posterior_entropy_over_links
        p_links = np.exp(entropy_over_links) / np.sum(np.exp(entropy_over_links))
        p_var = internal_state._causality_matrix(p_links, fill_diag=0).sum(axis=1)

        return p_var


    def _compute_sign_change_prob(self, sensory_state):
        if not self._acting:
            return np.ones(2) / 2

        effect_variables = np.arange(self._K) != self._current_action[0]
        changes = np.abs(sensory_state.s_alt[effect_variables])

        # Parameters that describes the attention given to the effect variables, default is uniform
        attention_params = np.ones(changes.shape) / changes.size

        abs_change_history = np.abs(sensory_state.obs_alt[:, effect_variables])

        # Large when current change is small compared to historic change
        # Represents p(keep sign)
        p = np.sum(changes * attention_params) / (np.sum(changes * attention_params) + 1e-5* abs_change_history.mean())

        # Standardise it so that first is p-negative and second is p-positive
        if self._current_action[1] < 0:
            p_std = np.array([p, 1-p])
        else:
            p_std = np.array([1-p, p])
        
        return p_std


    def _flatten_action(self, action):
        return np.where(self._poss_actions == action[1])[0]
        




