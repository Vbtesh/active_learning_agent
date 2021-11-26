import numpy as np
from classes.action_states import Experience_AS

class Experience_2D_AS(Experience_AS):
    def __init__(self, N, K, behaviour, possible_actions, policy_funcs, epsilon, max_acting_time):
        super().__init__(N, K, behaviour, possible_actions, policy_funcs, epsilon, self._information_gain_func)

        # Dimensions can be Value X Length, or Value X Length X Observations
        # Action values can be relative gain, absolue gain, change in variables
        self._max_acting_time = max_acting_time
        self._current_action_values = None

        # Think about splitting the function to allow different updates
        ## Discrete categorical space one learning rate applied only to the action taken
        ## Information (relative, absolute) or change
    

    def _action_values_update(self, action, action_len, action_obs, external_state, sensory_state, internal_state):
        # Compute information gained between beginning of action and current point in time 
        posterior_entropy = internal_state.posterior_entropy
        
        rollback = action_len + action_obs
        internal_state.rollback(rollback)
        entropy_at_action_start = internal_state.posterior_entropy

        information_gained = (entropy_at_action_start - posterior_entropy) / entropy_at_action_start

        # Learning rates
        action_x = action[1]
        action_y = action_len - 1

        x = self._poss_actions.size
        y = self._max_acting_time

        dist_x = (x - np.abs(np.arange(x) - action_x)).reshape((x, 1))
        dist_y = (y - np.abs(np.arange(y) - action_y)).reshape((1, y))
        grid = np.zeros((x, y))
        aggregate_dist = grid + dist_x + dist_y
        learning_rates = np.exp(aggregate_dist) / np.exp(aggregate_dist).sum()

        # Update action values
        updated_action_values = self._current_action_values + learning_rates * (information_gained - self._current_action_values)
        self._current_action_values = updated_action_values
        
        return updated_action_values


        

