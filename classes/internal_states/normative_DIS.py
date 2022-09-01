from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np

# Normative discrete agent
class Normative_DIS(Discrete_IS):
    def __init__(self, N, K, links, dt, theta, sigma, generate_sample_space=True, sample_params=False, prior_param=None, smoothing=False):
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


        # Collect observations to recompute mus
        self._obs_history = [None for _ in range(self._N+1)]
        self._obs_history[0] = np.zeros(self._K)


        
    # Update rule
    def _update_rule(self, sensory_state, action_state):
        intervention = action_state.a
        obs = sensory_state.s

        # Likelihood of observed the new values given the previous values for each model
        likelihood_per_var = stats.norm.logpdf(obs, loc=self._mus, scale=self._sigma*np.sqrt(self._dt)) # Compute probabilities

        # Normalisation step
        likelihood_log = likelihood_per_var - np.amax(likelihood_per_var, axis=0)
        #likelihood_per_var_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum(axis=0)
        
        ## If intervention, the probability of observing the new values is set to 1
        if isinstance(intervention, tuple):
            #likelihood_per_var_norm[:, intervention[0]] = 1
            likelihood_log[:, intervention[0]] = 0
 
        # Compute and normalise probabilities of each model given the previous and new values
        #likelihood_to_prop = likelihood_per_var_norm.prod(axis=1)
        #likelihood_over_models = likelihood_to_prop / likelihood_to_prop.sum()
        
        # Posterior params is the log likelihood of each model given the data
        ## The where argument is a problem, it makes it so models that are so unlikely that their probability is essentially 0 don't have their log likelihood penalised
        ## Cannot achieve numerical stability without it
        LL = likelihood_log.sum(axis=1)
        #LL = np.log(likelihood_over_models, where=likelihood_over_models!=0)
        log_posterior = self._posterior_params + LL
        #log_posterior = self._posterior_params + np.log(likelihood_over_models)

        #LL_l = self._likelihood(LL)
        #LL_c = self._likelihood(log_likelihood)

        #LL_lp = self._likelihood(LL + self._posterior_params)
        #LL_cp = self._likelihood(log_posterior)

        # Update mus
        self._update_mus(obs)

        # Return log_posterior over models
        return log_posterior


    # Background methods
    ## Prior initialisation specific to model:
    def _local_prior_init(self):
        if len(self._prior_params.shape) == 2:
            prior = self._links_to_models(self._prior_params)
            self._prior_params = np.log(prior)

        # Compute initial attractor
        self._mus = self._attractor_mu(self._obs_history[self._n])

    # Update attractors for all models
    def _update_mus(self, obs):
        self._mus = self._attractor_mu(obs)
        self._obs_history[self._n+1] = obs


    def _attractor_mu(self, obs): 
        att_mu =  obs @ self._sample_space_as_mat 
        self_mu =  -1 * obs * (np.abs(obs) / 100)
        mu_squeezed = np.squeeze(att_mu + self_mu)
        mus = obs + (mu_squeezed - obs) * self._theta * self._dt
        return mus


    def mus_model(self, graph, idx=None):
        if idx:
            graph_idx = idx
        else:
            graph_idx = np.argmax(np.where((self._sample_space == graph).all(axis=1))[0])
        graph_hist = np.zeros((len(self._mus_history), self._K))

        att_mu = np.zeros((len(self._mus_history), self._K))
        self_mu =  np.zeros((len(self._mus_history), self._K))
        for i in range(len(self._mus_history)):
            graph_hist[i,:] = self._mus_history[i][graph_idx]
            att_mu[i,:] = self._mu_att[i][graph_idx]
            self_mu[i,:] = self._self_att[i]

        return graph_hist, att_mu, self_mu

