from classes.internal_states.internal_state import Continuous_IS
from scipy import stats
import numpy as np

from methods.smoothing import entropy

# Local computation discrete agent
class Local_computations_omniscient_CIS(Continuous_IS):
    def __init__(self, N, K, prior_params, dt, theta, sigma, sample_params=True, init_obs=None):
        super().__init__(N, K, prior_params, dt, theta, sigma, self._update_rule, sample_params=sample_params)

        self._prior_params = prior_params
        self._init_priors()

        if init_obs:
            self._prev_obs = init_obs
        else:
            self._prev_obs = np.zeros(self._K)


    def _update_rule(self, sensory_state, intervention=None):
        obs = sensory_state.s

        new_params = np.zeros(self._prior_params.shape)

        mu_self = obs * (1 - np.abs(obs) / 100)

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
                    
                    sd = (self._prev_obs[i]**2 / self._dt + 1 / sd_prev)**(-1/2)
                    mean = sd**2 * (self._prev_obs[i]*(obs[j] - mu_self[j]) / self._dt + mean_prev / sd_prev**2)

                    new_params[idx, :] = [mean, sd]
                    idx += 1    

        self._prev_obs = obs    

        return new_params

    
    def _argmax(self):
        return self._posterior_params[:, 0]

    
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
        return np.sum(stats.norm,entropy(scale=parameters[:, 1]))


    def _pdf(self, obs, log=False):
        means = self._posterior_params[:, 0]
        sds = self._posterior_params[:, 1]
        if log:
            return stats.norm.logpdf(obs, loc=means, scale=sds)
        else:
            return stats.norm.pdf(obs, loc=means, scale=sds)

    
    def _link_pdf(self, link_idx, link_value, log=False):
        mean = self._posterior_params[link_idx, 0]
        sd = self._posterior_params[link_idx, 1]
        if log:
            return stats.norm.logpdf(link_value, loc=mean, scale=sd)
        else:
            return stats.norm.pdf(link_value, loc=mean, scale=sd)

    
