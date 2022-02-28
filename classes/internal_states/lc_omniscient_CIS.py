from random import sample
from classes.internal_states.internal_state import Continuous_omniscient_IS
from scipy import stats
import numpy as np


# Local computations continuous agent
class Local_computations_omniscient_CIS(Continuous_omniscient_IS):
    def __init__(self, N, K, prior_params, links, dt, theta, sigma, sample_params=True, smoothing=0, init_obs=None):
        super().__init__(N, K, prior_params, links, dt, self._update_rule, sample_params=sample_params, smoothing=smoothing)

        # Sample parameter estimates
        if sample_params:
            # Sample key variables according to Davis, Rehder, Bramley (2018)
            self._theta = stats.gamma.rvs(100*theta, scale=1/100, size=1)
            self._sigma = stats.gamma.rvs(100*sigma, scale=1/100, size=1)
        else:
            # Assume perfect knowledge
            self._theta = theta
            self._sigma = sigma  

        self._prior_params = prior_params
        self._init_priors()

        if init_obs:
            self._prev_obs = init_obs
        else:
            self._prev_obs = np.zeros(self._K)


    def _update_rule(self, sensory_state, intervention=None):
        obs = sensory_state.s

        new_params = np.zeros(self._prior_params.shape)

        mu_self = self._prev_obs * (1 - np.abs(self._prev_obs) / 100)

        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    mean_prev = self._posterior_params[idx, 0]
                    sd_prev = self._posterior_params[idx, 1]

                    if isinstance(intervention, tuple):
                        if j == intervention[0]:
                            sd = sd_prev
                            mean = mean_prev
                            new_params[idx, :] = [mean, sd]
                            idx += 1
                            continue
                    
                    sd = (self._prev_obs[i]**2 / self._dt + 1 / sd_prev**2)**(-1/2)
                    in_between = self._prev_obs[i]*(obs[j] - mu_self[j])
                    mean = sd**2 * (self._prev_obs[i]*(obs[j] - mu_self[j]) / self._dt + mean_prev / sd_prev**2)

                    new_params[idx, :] = [mean, sd]
                    idx += 1    

        self._prev_obs = obs    

        return new_params

    
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

    
