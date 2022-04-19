from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np


# Causal event segmentation model

class causal_event_segmentation_DIS(Discrete_IS):
    def __init__(self, N, K, links, prior_param, dt, abs_bounds, ce_threshold=0.5, time_threshold = 15, guess=0.1, generate_sample_space=True, sample_params=False, smoothing=False):
        super().__init__(N, K, links, prior_param, dt, self._update_rule, generate_sample_space=generate_sample_space, sample_params=sample_params, smoothing=smoothing)

        self._bounds = abs_bounds

        # Threshold for event detection
        # Given by a percentage of the full range
        self._causal_event_threshold = ce_threshold * abs_bounds[1]

        # Time aspect
        self._time_threshold = time_threshold

        # Probability mass to be split between non causal event segmentation predictions
        self._guess = guess

        self._last_action = None
        self._last_action_len = None
        self._last_instant_action = None
        self._last_real_action = None
        self._last_obs = np.zeros(self._K)
        self._last_action_idx = 0




    def _update_rule(self, sensory_state, action_state):
        # Get current action from action state
        ## Beware: if not a_fit but a_real, this can lead to divergence because the model will interpret an intervention as normal data
        ## Maybe use a_real rather that a_fit
        intervention = action_state.a
        
        obs = sensory_state.s
        obs_alt = sensory_state.s_alt 
        

        # Action started but not learnable action
        # If fitting, check between fit and real action
        if action_state.realised:
            # If fitting, check between fit and real action
            if (not self._last_action and not action_state.a_real) or self._n == 0:
                self._last_obs = obs
                self._last_instant_action = action_state.a_real
                return self._posterior_params

            elif not self._last_action and action_state.a_real:
                # First action
                self._last_action_len = action_state.a_len_real      
                # Reset last action index
                self._last_action_idx = 0

                self._last_action = action_state.a_real

                self._last_action_idx += 1
                self._last_obs = obs
                self._last_instant_action = action_state.a_real
                

            elif self._last_action and action_state.a_real:
                if not self._last_instant_action:
                    # CHANGE OF ACTION
                    # Action length is
                    self._last_action_len = action_state.a_len_real      
                    # Reset last action index
                    self._last_action_idx = 0

                    self._last_action = action_state.a_real

                    self._last_action_idx += 1
                    self._last_obs = obs
                    self._last_instant_action = action_state.a_real
                        
                else:
                    self._last_action_idx += 1
                    self._last_obs = obs
                    self._last_instant_action = action_state.a_real
                        
        else:
            # If generating data
            if (not self._last_action and not intervention) or self._n == 0:
                self._last_obs = obs
                self._last_instant_action = intervention
                return self._posterior_params
            elif not self._last_action and intervention:
                # Action length is
                self._last_action_len = action_state.a_len     
                # Reset last action index
                self._last_action_idx = 0

                self._last_action = intervention
                self._last_instant_action = intervention
            elif self._last_action and intervention:
                if not self._last_instant_action:
                    # Action length is
                    self._last_action_len = action_state.a_len     
                    # Reset last action index
                    self._last_action_idx = 0

                    self._last_action = intervention
                    self._last_instant_action = intervention
    
        
        # Logic for updating
        # Compute power update coefficient
        delay = self._last_action_len
        power_coef = self._power_update_coef(delay)
        self._power_update_coef_history[self._n] = power_coef

        ## Get change
        ## Update evidence or values (posterior value)
        log_likelihood_per_link = np.zeros(self._posterior_params.shape)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    if j == self._last_action[0] or i != self._last_action[0]:
                        log_likelihood_per_link[idx, :] = np.zeros(self._L.size)
                        idx += 1
                        continue

                    
                    if np.abs(obs[j]) > self._causal_event_threshold:
                        if self._last_action_idx < self._time_threshold:
                            # Accumulate strong link evidence 
                            if obs[j] * obs[i] > 1:
                                # Positive link
                            else:
                                # Negative link
                        else:
                            # Accumulate weak link evidence
                            if obs[j] * obs[i] > 1:
                                # Positive link
                            else:
                                # Negative link
                    
                    
                    idx += 1
                    
        
        # Posterior params is the log likelihood of each model given the data
        log_posterior = self._posterior_params + log_likelihood_per_link

        # update mus
        self._last_action_idx += 1
        self._last_obs = obs

        if action_state.realised:
            self._last_instant_action = action_state.a_real
        else:
            self._last_instant_action = intervention

        return log_posterior
        

    
    # Background methods
    ## Prior initialisation specific to model:
    def _local_prior_init(self):
        self._summary_stats_history = np.zeros((self._prior_params.shape[0], self._N))





