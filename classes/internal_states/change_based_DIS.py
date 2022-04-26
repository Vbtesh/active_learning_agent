from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np


# Change based LC agent, assuming linear relationships between variables
# Free parameters:
## c: proportionality constant relating sampled summary stats to link values (optimal should be dt*theta)
## decay rate: rate at which attention decays (the decay is discounted according to the time unit which can be seconds or frames)
## sigma: std deviation of the likelihood function (will never change the MAP but will change the entropy, hence big impact on log likelihood)

# Choose update type:
## Proportional to change
## Proportional to cause variable

# Choose decay type:
## Exponential
## Sigmoid

class LC_linear_change_DIS(Discrete_IS):
    def __init__(self, N, K, links, dt, prop_const, hypothesis, decay_type, lh_var=1/10, decay_rate=0.65, generate_sample_space=True, sample_params=False, prior_param=None,  smoothing=False):
        super().__init__(N, K, links, dt, self._update_rule, generate_sample_space=generate_sample_space, sample_params=sample_params, prior_param=prior_param, smoothing=smoothing)

        self._c = 1 / prop_const
        self._sigma = lh_var**(1/2)

        if hypothesis == 'distance':
            self._calc_obs_stat = self._prop_distance
        elif hypothesis == 'cause_value':
            self._calc_obs_stat = self._prop_cause_value
        elif hypothesis == 'cause_effect_values':
            self._calc_obs_stat = self._cause_effect_values
        elif hypothesis == 'full_knowledge':
            self._calc_obs_stat = self._full_knowledge

        if decay_type == 'exponential':
            self._power_update_coef = self._exponential_decay
        elif decay_type == 'sigmoid':
            self._power_update_coef = self._sigmoid_decay

        self._decay_rate = decay_rate

        self._last_action = None
        self._last_action_len = None
        self._last_instant_action = None
        self._last_real_action = None
        self._last_obs = np.zeros(self._K)
        self._last_action_idx = 0

        self._power_update_coef_history = np.zeros(self._N)


    def _update_rule(self, sensory_state, action_state):
        # Get current action from action state
        ## Beware: if not a_fit but a_real, this can lead to divergence because the model will interpret an intervention as normal data
        ## Maybe use a_real rather that a_fit
        intervention = action_state.a
        
        obs = sensory_state.s
        obs_alt = sensory_state.s_alt 
        
        #### ACTION LOGIC FIT
        # Action started but not learnable action
        # If fitting, check between fit and real action
        #if action_state.realised:
        #    # If fitting, check between fit and real action
        #    if (not self._last_action and not action_state.a_real) or self._n == 0:
        #        self._last_obs = obs
        #        self._last_instant_action = action_state.a_real
        #        return self._posterior_params

        #    elif not self._last_action and action_state.a_real:
        #        # First action
        #        self._last_action_len = action_state.a_len_real      
        #        # Reset last action index
        #        self._last_action_idx = 0

        #        self._last_action = action_state.a_real

        #        if not intervention:
        #            self._last_action_idx += 1
        #            self._last_obs = obs
        #            self._last_instant_action = action_state.a_real
        #            return self._posterior_params

        #    elif self._last_action and action_state.a_real:
        #        if not self._last_instant_action:
        #            # CHANGE OF ACTION
        #            # Action length is
        #            self._last_action_len = action_state.a_len_real      
        #            # Reset last action index
        #            self._last_action_idx = 0

        #            self._last_action = action_state.a_real

        #            if not intervention:
        #                self._last_action_idx += 1
        #                self._last_obs = obs
        #                self._last_instant_action = action_state.a_real
        #                return self._posterior_params
        #        else:
        #            if not intervention:
        #                self._last_action_idx += 1
        #                self._last_obs = obs
        #                self._last_instant_action = action_state.a_real
        #                return self._posterior_params
        ### ACTION LOGIC REAL
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

                if not intervention:
                    self._last_action_idx += 1
                    self._last_obs = obs
                    self._last_instant_action = action_state.a_real
                    return self._posterior_params

            elif self._last_action and action_state.a_real:
                if not self._last_instant_action:
                    # CHANGE OF ACTION
                    # Action length is
                    self._last_action_len = action_state.a_len_real      
                    # Reset last action index
                    self._last_action_idx = 0

                    self._last_action = action_state.a_real

                    if not intervention:
                        self._last_action_idx += 1
                        self._last_obs = obs
                        self._last_instant_action = action_state.a_real
                        return self._posterior_params
                else:
                    if not intervention:
                        self._last_action_idx += 1
                        self._last_obs = obs
                        self._last_instant_action = action_state.a_real
                        return self._posterior_params
    
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

                    # Compute summary stat
                    summary_stat = self._calc_obs_stat(obs_alt[j], self._last_obs[i], self._last_obs[j])
                        
                    # Control divergence and weird interventions
                    if np.abs(summary_stat) == np.inf:
                        log_likelihood_per_link[idx, :] = np.zeros(self._L.size)
                        idx += 1
                        continue

                    # Likelihood of observed the new values given the previous values for each model
                    log_likelihood = power_coef * stats.norm.logpdf(summary_stat, loc=self._L, scale=self._sigma)
                    # Normalisation step
                    likelihood_log = log_likelihood - np.amax(log_likelihood)
                    likelihood_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum()
                    log_likelihood_per_link[idx, :] = np.log(likelihood_norm)
                    self._summary_stats_history[idx, self._n] = summary_stat
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

    # Summary statistics               
    ## Proportional to cause value
    def _prop_cause_value(self, change_effect, cause, effect):
        if np.abs(cause) == 0 or np.abs(change_effect) > 20:
            return np.inf

        return self._c * change_effect / cause

    ## Additive cause and effect values
    def _cause_effect_values(self, change_effect, cause, effect):
        if np.abs(cause) == 0 or np.abs(change_effect) > 20:
            return np.inf

        return self._c * change_effect / cause + effect / cause

    ## Full knowledge
    def _full_knowledge(self, change_effect, cause, effect):
        if np.abs(cause) == 0 or np.abs(change_effect) > 20:
            return np.inf

        return self._c * change_effect / cause + (effect * (np.abs(effect) / 100) )/ cause
    
    ## Proportional to distance
    ### NOT FUNCTIONAL: Requires additional logic as it fails for negative links
    def _prop_distance(self, change_effect, cause, effect):
        if np.abs(cause - effect) == 0 or np.abs(change_effect) > 20:
            return np.inf

        return self._c * (change_effect / (cause - effect))
        

    # Power update coefficients
    ## Exponential decay
    def _exponential_decay(self, delay):
        return self._decay_rate**self._last_action_idx

    ## Sigmoid decay
    def _sigmoid_decay(self, delay):
        return 1 / (1 + self._decay_rate**(- self._last_action_idx + delay))



