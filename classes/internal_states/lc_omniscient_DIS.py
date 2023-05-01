from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np

# Local computation discrete agent
class Local_computations_omniscient_DIS(Discrete_IS):
    def __init__(self, N, K, links, dt, theta, sigma, evidence_weight=1, generate_sample_space=True, sample_params=False, prior_param=None, smoothing=False):
        super().__init__(N, K, links, dt, self._update_rule, generate_sample_space=generate_sample_space, sample_params=sample_params, prior_param=prior_param, smoothing=smoothing)

        # Sample parameter estimates
        if sample_params:
            # Sample key variables according to Davis, Rehder, Bramley (2018)
            self._theta = stats.gamma.rvs(100*theta, scale=1/100, size=1)
            self._sigma = stats.gamma.rvs(100*sigma, scale=1/100, size=1)
        else:
            # Assume perfect knowledge
            self._theta = theta
            self._sigma = sigma  
        
        # Evidence weight 
        self._evidence_weight = evidence_weight

        self._prior_param = prior_param

        # Special parameters for faster computations
        self._links_lc_updates = np.tile(links.reshape((links.size, 1)), 3).T

        # Define own attractor mu, should be mu for each given the other two
        self._mus = self._attractor_mu(np.zeros(self._K))
        self._mus_history = [None for i in range(self._N)]
        self._mus_history[0] = self._mus

        

    def _update_rule(self, sensory_state, action_state):
        intervention = action_state.a
        obs = sensory_state.s

        # Logic for updating
        log_likelihood_per_link = np.zeros(self._posterior_params.shape)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    # Likelihood of observed the new values given the previous values for each model
                    log_likelihood = self._evidence_weight * stats.norm.logpdf(obs[j], loc=self._mus[idx, :], scale=self._sigma*np.sqrt(self._dt))
                    # Normalisation step
                    likelihood_log = log_likelihood - np.amax(log_likelihood)
                    #likelihood_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum()

                    ### If intervention, the probability of observing the new values is set to 1
                    #if isinstance(intervention, tuple):
                    #    if j == intervention[0]:
                    #        likelihood_norm[:] = 1

                    #log_likelihood_per_link[idx, :] = np.log(likelihood_norm)

                    # If intervention, the probability of observing the new values is set to 1
                    if isinstance(intervention, tuple):
                        if j == intervention[0]:
                            likelihood_log[:] = 0

                    log_likelihood_per_link[idx, :] = likelihood_log
                    idx += 1
        
        # Posterior params is the log likelihood of each model given the data
        log_posterior = self._posterior_params + log_likelihood_per_link

        # update mus
        self._update_mus(obs)

        return log_posterior

    
    # Background methods
    ## Prior initialisation specific to model:
    def _local_prior_init(self):
        self._prior_params = np.log(self._prior_params)
        self._mus = self._mus_history[self._n]
    
    # Mus
    def _update_mus(self, obs):
        self._mus_history[self._n] = self._mus
        self._mus = self._attractor_mu(obs)

    
    def _attractor_mu(self, obs):
        mu_self = obs * (1 - np.abs(obs) / 100)
        mu_att = obs.reshape((self._K, 1)) * self._links_lc_updates

        mus = np.zeros((self._K**2 - self._K, self._L.size))
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    mus[idx, :] = obs[j] + (mu_att[i, :] + mu_self[j] - obs[j]) * self._dt * self._theta
                    idx += 1

        return mus

