import numpy as np
from classes.action_states.action_state import Experience_AS

class Experience_3D_AS(Experience_AS):
    def __init__(self, N, K, behaviour, epsilon, possible_actions, policy_funcs, time_unit, max_acting_time, max_obs_time, experience_measure, prior_action_values, discount=0.9):        
        super().__init__(N, K, behaviour, epsilon, possible_actions, policy_funcs, self._action_values_update)

        # Pick experience gained function
        if experience_measure == 'information':
            self._compute_experience_gained = self._compute_information_gained
        elif experience_measure == 'change':
            self._compute_experience_gained = self._compute_change_generated
            self._change_discount = discount

        # Dimensions can be Value X Length, or Value X Length X Observations
        # Action values can be relative gain, absolue gain, change in variables
        self._max_acting_time = max_acting_time
        self._max_obs_time = max_obs_time

        # Time unit
        self._dt = time_unit
        
        # NEED TO BE SOMETHING: Prior over action values that can carry over from trial to trial
        self._current_action_values = prior_action_values

        # Think about splitting the function to allow different updates
        ## Discrete categorical space one learning rate applied only to the action taken
        ## Information (relative, absolute) or change

        ### FINAL STATE
        ## always be 3d, average over observation for 2D.
        ## Log likelihood can be compared in 3d to tree search as both give distribution over all datapoints (if unit is second!)
    
    def _compute_information_gained(self, action, rollback, external_state, sensory_state, internal_state):
        # Compute information gained between beginning of action and current point in time 
        posterior_entropy = internal_state.posterior_entropy
        internal_state.rollback(rollback)
        entropy_at_action_start = internal_state.posterior_entropy

        information_gained = (entropy_at_action_start - posterior_entropy) / entropy_at_action_start
        return information_gained


    def _compute_change_generated(self, action, rollback, external_state, sensory_state, internal_state):
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


    def _action_values_update(self, action, action_len, action_obs, external_state, sensory_state, internal_state):
        if not action:
            return self._current_action_values
           
        rollback = action_len + action_obs
        experience_gained = self._compute_experience_gained(action, rollback, external_state, sensory_state, internal_state)
        
        # Learning rates
        action_x = action[1]
        action_y = action_len - 1
        action_z = action_obs - 1

        x = self._poss_actions.size
        y = self._max_acting_time
        z = self._max_obs_time

        dist_x = (x - np.abs(np.arange(x) - action_x)).reshape((x, 1, 1))
        dist_y = (y - np.abs(np.arange(y) - action_y)).reshape((1, y, 1))
        dist_z = (z - np.abs(np.arange(z) - action_z)).reshape((1, 1, z))
        grid = np.zeros((x, y, z))
        aggregate_distance = grid + dist_x + dist_y + dist_z
        learning_rates = np.exp(aggregate_distance) / np.exp(aggregate_distance).sum()

        # Update action values
        updated_action_values = self._current_action_values + learning_rates * (experience_gained - self._current_action_values)
        self._current_action_values = updated_action_values
        
        return updated_action_values


        

