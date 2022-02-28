import numpy as np
from scipy import stats
from classes.action_states.action_state import Experience_AS


class Experience_conti_3D_AS(Experience_AS):
    def __init__(self, N, K, behaviour, epsilon, possible_actions, policy_funcs, time_unit, max_acting_time, max_obs_time, experience_measure, prior_gaussian_params, learning_param, discount=0.9):        
        super().__init__(N, K, behaviour, epsilon, possible_actions, policy_funcs, self._distribution_update)

        # Pick experience gained function
        if experience_measure == 'information':
            self._compute_experience_gained = self._compute_information_gained
            self._learning_rate = self._learning_rate_information
        elif experience_measure == 'change':
            self._compute_experience_gained = self._compute_change_generated
            self._change_discount = discount
            self._learning_rate = self._learning_rate_change

        # Dimensions can be Value X Length, or Value X Length X Observations
        # Action values can be relative gain, absolue gain, change in variables
        self._num_values = self._poss_actions.size
        self._max_acting_time = max_acting_time
        self._max_obs_time = max_obs_time

        # Aggregate variables
        self._dims = (self._num_values, self._max_acting_time, self._max_obs_time)
        self._num_actions = self._num_values * self._max_acting_time * self._max_obs_time

        # Time unit
        self._dt = time_unit

        # Action sample space generation
        self._action_space = self._generate_action_space()

        # Learning params
        self._learning_param = learning_param

        # Prior Gaussian parameters
        self._prior_params = prior_gaussian_params

        # Book keeping
        self._current_params = prior_gaussian_params
        self._params_history = [None for i in range(self._N+1)]
        self._params_history[0] = self._current_params
        self._learning_rate_history = np.zeros(self._N+1)

        self._current_distribution = self._discretise_distribution(self._prior_params)

        

    def _distribution_update(self, action, action_len, action_obs, sensory_state, internal_state):
        if not action:
            return self._current_distribution
           
        rollback = int(action_len) + int(action_obs)
        experience_gained = self._compute_experience_gained(action, rollback, sensory_state, internal_state)
        
        # Aggregate action data
        action_data = np.array([np.abs(action[1]), int(action_len*self._dt), int(action_obs*self._dt)])

        # Learning rates from experience_gained
        learning_rate = self._learning_rate(experience_gained)

        # Collect mus
        mus = self._current_params[0]
        
        updated_mus = mus + learning_rate * (action_data - mus)

        new_params = (updated_mus, self._current_params[1])
        
        # Record history
        self._params_history[self._n] = self._current_params
        self._learning_rate_history[self._n] = learning_rate

        # Update current parameters
        self._current_params = new_params
        self._current_distribution = self._discretise_distribution(new_params)

        return self._current_distribution
        

    # Takes the experience gained and return a learning rate between 0 and 1
    def _learning_rate_information(self, information_gained):
        #return self._learning_param * information_gained
        return self._learning_param

    def _learning_rate_change(self, change_observed):
        #return self._learning_param * change_observed / 10
        return self._learning_param


    def _compute_information_gained(self, action, rollback, sensory_state, internal_state):
        # Compute information gained between beginning of action and current point in time 
        posterior_entropy = internal_state.posterior_entropy
        internal_state.rollback(rollback)
        entropy_at_action_start = internal_state.posterior_entropy

        information_gained = (entropy_at_action_start - posterior_entropy) / entropy_at_action_start
        return information_gained


    def _compute_change_generated(self, action, rollback, sensory_state, internal_state):
        if not action:
            mask = None
        else:
            mask = action[0]

        selected_variables = np.logical_not(np.arange(self._K) == mask)

        changes = sensory_state.obs_alt[-rollback:, selected_variables]
        
        abs_changes = np.abs(changes).mean(axis=1)

        # Repeat on one second scale (5)
        rate = 5
        seconds = np.arange(abs_changes.size / rate)
        remain = rate - abs_changes.size % rate
        reps = np.repeat(seconds, rate)
        reps_clean = reps[:-int(remain)] if remain < rate else reps

        # Discount factors
        discounts = self._change_discount**reps_clean

        discounted_changes = np.sum(abs_changes * discounts)
        
        return discounted_changes


    def _discretise_distribution(self, params):
        mus = params[0]
        cov = params[1]

        probs = np.zeros((self._num_actions))
        
        for i in range(self._num_actions):
            probs[i] = stats.multivariate_normal.pdf(self._action_space[i, :], mean=mus, cov=cov)
        
        probs_norm = probs / probs.sum()

        return probs_norm.reshape(self._dims)


    def _generate_action_space(self):
        sample_space = np.zeros((self._num_actions, len(self._dims)))
        
        idx = 0
        for i in self._poss_actions:
            for j in range(self._max_acting_time):
                for k in range(self._max_obs_time):
                    sample_space[idx, :] = [i, j, k]
                    idx += 1

        return sample_space
    