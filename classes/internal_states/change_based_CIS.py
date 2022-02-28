from classes.internal_states.internal_state import Continuous_omniscient_IS
from scipy import stats
import numpy as np

# Change based LC agent, assuming linear relationships between variables
class LC_linear_change_CIS(Continuous_omniscient_IS):
    def __init__(self, N, K, prior_params, links, dt, threshold, sample_params=True, smoothing=False):
        super().__init__(N, K, prior_params, links, dt, self._update_rule, sample_params=sample_params, smoothing=smoothing)

        self._data_threshold = threshold
        
        self._prior_params = prior_params
        self._init_priors()

        self._last_action = None
        self._last_obs = np.zeros(self._K)
        self._last_action_idx = 0


    def _update_rule(self, sensory_state, intervention=None):
        if not intervention and not self._last_action:
            return self._posterior_params

        if intervention != self._last_action:
            self._last_action = intervention
            self._last_action_idx = 0

        obs = sensory_state.s
        inter_obs = sensory_state.s[self._last_action[0]]
        other_obs = sensory_state.s[np.arange(self._K) != self._last_action[0]]

        obs_alt = sensory_state.s_alt  
        inter_obs_alt = sensory_state.s_alt[self._last_action[0]] 
        other_obs_alt = sensory_state.s_alt[np.arange(self._K) != self._last_action[0]]

        # Logic for updating
        ## Get change
        ## Update evidence or values (posterior value)
        ## Transform into prob
        log_posterior = 0

        self._last_action_idx += 1
        self._last_obs = obs
        return log_posterior

    
    # Background methods
    

