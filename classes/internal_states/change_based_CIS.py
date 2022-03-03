from classes.internal_states.internal_state import Continuous_IS
from scipy import stats
import numpy as np

# Change based LC agent, assuming linear relationships between variables

# Choose update type:
## Proportional to change
## Proportional to cause variable

# Choose decay type:
## Exponential
## Sigmoid

class LC_linear_change_CIS(Continuous_IS):
    def __init__(self, N, K, prior_params, links, dt, prop_const, variance, hypothesis, decay_type, decay_rate, sample_params=True, smoothing=False):
        super().__init__(N, K, prior_params, links, dt, self._update_rule, sample_params=sample_params, smoothing=smoothing)

        self._c = 1 / prop_const
        self._sigma = variance**(1/2)

        self._prior_params = prior_params
        self._init_priors()

        if hypothesis == 'distance':
            self._calc_obs_stat = self._prop_distance
        elif hypothesis == 'cause_value':
            self._calc_obs_stat = self._prop_cause_value
        elif hypothesis == 'full_knowledge':
            self._calc_obs_stat = self._full_knowledge

        if decay_type == 'exponential':
            self._power_update_coef = self._exponential_decay
        elif decay_type == 'sigmoid':
            self._power_update_coef = self._sigmoid_decay

        self._decay_rate = decay_rate

        self._last_action = None
        self._last_obs = np.zeros(self._K)
        self._last_action_idx = 0

        self._summary_stats_history = np.zeros((self._prior_params.shape[0], self._N))
        self._power_update_coef_history = np.zeros(self._N)


    def _update_rule(self, sensory_state, action_state):
        intervention = action_state.a

        if not intervention and not self._last_action:
            return self._posterior_params

        if intervention and (intervention != self._last_action):
            self._last_action = intervention
            self._last_action_idx = 0

        obs = sensory_state.s
        obs_alt = sensory_state.s_alt 
        
        new_params = np.zeros(self._prior_params.shape)

        # Logic for updating
        # Compute power update coefficient
        delay = action_state.a_len
        power_coef = self._power_update_coef(delay)
        self._power_update_coef_history[self._n] = power_coef

        ## Get change
        ## Update evidence or values (posterior value)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    mean_prev = self._posterior_params[idx, 0]
                    sd_prev = self._posterior_params[idx, 1]

                    if j == self._last_action[0] or i != self._last_action[0]:
                        sd = sd_prev
                        mean = mean_prev
                        new_params[idx, :] = [mean, sd]
                        idx += 1
                        continue
                    
                    # Compute summary stat
                    summary_stat = self._calc_obs_stat(obs_alt[j], self._last_obs[i], self._last_obs[j])
                    # Control divergence
                    if np.abs(summary_stat) == np.inf:
                        sd = sd_prev
                        mean = mean_prev
                        new_params[idx, :] = [mean, sd]
                        idx += 1
                        continue
                    
                
                    # Compute posterior parameters
                    sd = (power_coef / self._sigma**2 + 1 / sd_prev**2)**(-1/2)
                    mean = sd**2 * (summary_stat * power_coef / self._sigma**2 + mean_prev / sd_prev**2)

                    new_params[idx, :] = [mean, sd]

                    self._summary_stats_history[idx, self._n] = summary_stat
                    idx += 1  
    
        self._last_action_idx += 1
        self._last_obs = obs

        return new_params
        

    
    # Background methods
    # Summary statistics               
    ## Proportional to distance
    def _prop_distance(self, change_effect, cause, effect):
        return self._c * (change_effect / (cause - effect))

    ## Proportional to cause value
    def _prop_cause_value(self, change_effect, cause, effect):
        return self._c * change_effect / cause

    ## Full knowledge
    def _full_knowledge(self, change_effect, cause, effect):
        return self._c * change_effect / cause + effect / cause
        

    # Power update coefficients
    ## Exponential decay
    def _exponential_decay(self, delay):
        return self._decay_rate**self._last_action_idx

    ## Sigmoid decay
    def _sigmoid_decay(self, delay):
        return 1 / (1 + self._decay_rate**(- self._last_action_idx + delay))

    
    # Generic Continuous functions
    def _argmax(self):
        return np.round(self._posterior_params[:, 0], 2)

    
    def _sample_distribution(self, size=1):
        means = self._posterior_params[:, 0]
        sds = self._posterior_params[:, 1]

        if size == 1:
            return stats.norm.rvs(loc=means, scale=sds)
        else:
            samples = np.zeros((size, means.size))
            for i in range(size):
                samples[i, :] = stats.norm.rvs(loc=means, scale=sds)
            return samples


    def _entropy_distribution(self, parameters):
        return stats.norm.entropy(scale=parameters[:, 1])


    def _posterior_pmf(self, params):
        means = params[:, 0].reshape((params.shape[0], 1))
        sds = params[:, 1].reshape((params.shape[0], 1))

        discrete_pmf = stats.norm.cdf(self._L + self._interval, loc=means, scale=sds) - stats.norm.cdf(self._L - self._interval, loc=means, scale=sds)
        discrete_pmf_norm = discrete_pmf / discrete_pmf.sum(axis=1).reshape((means.shape))

        return discrete_pmf_norm


    def _pdf(self, obs):
        discrete_PMF = self.posterior
        logic = np.tile(self._L, (discrete_PMF.shape[0], 1)) == obs.reshape((discrete_PMF.shape[0], 1))

        return discrete_PMF[logic]

    
    def _link_pdf(self, link_idx, link_value):
        discrete_PMF = self.posterior
        link_discrete_PMF = discrete_PMF[link_idx, :]
        logic = np.squeeze(self._L == link_value)
        
        return link_discrete_PMF[logic][0]

    

