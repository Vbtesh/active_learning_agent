from math import log
import numpy as np
import pandas as pd
from scipy import stats


# Main Internal state class
class Internal_state():
    def __init__(self, N, K, prior_params, update_func, update_func_args=[]):
        self._N = N
        self._n = 0
        self._K = K

        self._prior_params = prior_params
        self._init_priors()

        # p(i_t|s_t, i_t-1): must be a function of sensory states and the last action
        self._p_i_g_s_i = update_func
        self._p_i_g_s_i_args = update_func_args

        self._realised = False
        

    
    # General update function that all internal state object must satisfy
    ## Parameters can change but there must always be current set of posterior parameters & a history of the parameters at each time step
    def update(self, sensory_state, intervention=None):
        self._posterior_params_history[self._n] = self._posterior_params

        self._posterior_params = self._p_i_g_s_i(sensory_state, intervention, *self._p_i_g_s_i_args)
        
        self._n += 1
        
        if self._realised:
            # Update history
            self._log_likelihood_history[self._n] = self._log_likelihood

            judgement_log_prob = 0
            j_data = self._judgement_data[self._n-1, :]

            if np.sum(np.isnan(j_data) != True) > 0:
                link_idx = np.argmax(np.isnan(j_data) != True)
                link_value = j_data[link_idx]

                judgement_log_prob = self.posterior_PMF_link(link_idx, link_value, log=True)

                self._judgement_current[link_idx] = link_value

            elif self._n == self._N:
                # Fit final judgement
                judge_diff = self._judgement_current != self._judgement_final

                
                for i in range(judge_diff.size):
                    if judge_diff[i]:
                        link_idx = i
                        link_value = self._judgement_final[link_idx]

                        judgement_log_prob += self.posterior_PMF_link(link_idx, link_value, log=True)

                        self._judgement_current[link_idx] = link_value

                print('Fiiting final judgement:', judgement_log_prob)

            self._log_likelihood += judgement_log_prob

            return judgement_log_prob
        

    def load_judgement_data(self, judgement_data, final_judgement):
        self._judgement_data = judgement_data
        self._judgement_final = final_judgement

        self._judgement_current = np.empty(self._K**2 - self._K)
        self._judgement_current[:] = np.nan

        self._log_likelihood = 0
        self._log_likelihood_history = np.zeros(self._N+1)

        self._realised = True

    
    # Roll back internal state by a given number of step
    ## Used mostly for action selection
    def rollback(self, back=np.Inf):
        if back > self._N or back > self._n:
            self._n = 0
            self._init_priors()
        else:
            self._n -= back

            self._posterior_params = self._posterior_params_history[self._n]
            # Reset Action values, seq and planned action from n to N
            for n in range(self._n+1, self._N):
                self._posterior_params_history[n] = None
                

    # Utility functions
    def _init_priors(self):
        self._posterior_params = self._prior_params
        self._posterior_params_history = [None for i in range(self._N)]
        self._posterior_params_history[0] = self._prior_params


    def _causality_matrix(self, link_vec, fill_diag=1):
        num_var = int((1 + np.sqrt(1 + 4*len(link_vec))) / 2)
        causal_mat = fill_diag * np.ones((num_var, num_var))

        idx = 0
        for i in range(num_var):
            for j in range(num_var):
                if i != j:
                    causal_mat[i, j] = link_vec[idx] 
                    idx += 1

        return causal_mat

    
    def _causality_vector(self, link_mat, dim2=None):
        s = link_mat.shape[0]**2 - link_mat.shape[0]

        if dim2:
            causal_vec = np.zeros((s, dim2))
        else:
            causal_vec = np.zeros(s)

        idx = 0
        for i in range(link_mat.shape[0]):
            for j in range(link_mat.shape[1]):
                if i != j:
                    if dim2:
                        causal_vec[idx, :] = link_mat[i, j]
                    else:
                        causal_vec[idx] = link_mat[i, j]
                    idx += 1

        return causal_vec


# Internal state using a discrete probability distribution to represent the external states
class Discrete_IS(Internal_state):
    def __init__(self, N, K, prior_params, links, dt, theta, sigma, update_func, update_func_args=[], sample_params=True):
        super().__init__(N, K, prior_params, update_func, update_func_args=update_func_args)

        self._num_links = len(links)
        self._dt = dt
        
        # Sample parameter estimates
        if sample_params:
            # Sample key variables according to Davis, Rehder, Bramley (2018)
            self._theta = stats.gamma.rvs(100*theta, scale=1/100, size=1)
            self._sigma = stats.gamma.rvs(100*sigma, scale=1/100, size=1)

            self._L = np.zeros(len(links))
            for i, l in enumerate(links):
                if l < 0:
                    self._L[i] = -1 * stats.gamma.rvs(100*np.abs(l), scale=1/100)
                elif l == 0:
                    self._L[i] = 0
                else:
                    self._L[i] = stats.gamma.rvs(100*l, scale=1/100)
        else:
            # Assume perfect knowledge
            self._theta = theta
            self._sigma = sigma
            self._L = links

        # Build representational spaces
        self._sample_space = self._build_space(links) # Sample space as set of vectors with link values
        self._indexed_space = self._build_space(np.arange(len(links))).astype(int) # Same as above but link values are indices
        self._sample_space_as_mat = self._build_space(links, as_matrix=True) # Sample space as a set of causality matrices, i.e. one for each model

        #self.causes_idx = [self._map_cause_to_effect(i, K**2-K) for i in range(K)] # Internal variables for converting from distribution over models to distribution over links

    # Properties
    @property
    def posterior(self):
        return self._likelihood(self._posterior_params)

    @property
    def posterior_over_links(self):
        if len(self.posterior.shape) == 1:
            return self._models_to_links(self.posterior)
        else:
            return self.posterior

    @property
    def posterior_over_models(self):
        if len(self.posterior.shape) == 1:
            return self.posterior
        else:
            return self._links_to_models(self.posterior) 

    @property
    def map(self):
        graph_idx = np.argmax(self.posterior_over_models)
        return self._sample_space[graph_idx]

    @property
    def posterior_entropy(self):
        return self._entropy(self.posterior)
    
    @property
    def entropy_history(self):
        posterior_history = self._likelihood(self._posterior_params_history)
        return self._entropy(posterior_history)


    # Samples the posterior, the number of samples is given by the size parameter
    def posterior_sample(self, size=1, uniform=False, as_matrix=False):
        if uniform:
            graph_idx = np.random.choice(np.arange(self._sample_space.shape[0]), size=size)
        else:
            graph_idx = np.random.choice(np.arange(self._sample_space.shape[0]), size=size, p=self.posterior)
            
        if as_matrix:
            return self._sample_space_as_mat[graph_idx].squeeze()
        else:
            return self._sample_space[graph_idx].squeeze()

    # PMF of the posterior for a given graph
    def posterior_PMF(self, graph, log=False):
        if not log:
            return self.posterior_over_models[np.where((self._sample_space == graph).all(axis=1))[0]]
        else:  
            return np.log(self.posterior_over_models[np.where((self._sample_space == graph).all(axis=1))[0]])

    # PMF of the posterior for a given link
    def posterior_PMF_link(self, link_idx, link_value, log=False):
        value_idx = np.squeeze(np.where(self._L == link_value)[0])
        if not log:
            prob = self.posterior_over_links[link_idx, value_idx]
            return prob
        else:
            log_prob = np.log(self.posterior_over_links[link_idx, value_idx])
            return log_prob


    # Background methods for likelihood and sampling for discrete distributions
    def _likelihood(self, log_likelihood):
        if isinstance(log_likelihood, list):
            LL = np.array(log_likelihood[0:self._n+1])
            # Case where likelihood is 2 dimensional (LC) and we want the whole history
            if len(LL.shape) == 3: 
                LL_n = LL - np.amax(LL, axis=2).reshape((LL.shape[0], LL.shape[1], 1))
                return np.squeeze(np.exp(LL_n) / np.sum(np.exp(LL_n), axis=2).reshape((LL.shape[0], LL.shape[1], 1)))
        else:
            LL = log_likelihood

        if len(LL.shape) == 1:
            LL = LL.reshape(1, LL.shape[0])
        LL_n = LL - np.amax(LL, axis=1).reshape(LL.shape[0], 1)
        return np.squeeze(np.exp(LL_n) / np.sum(np.exp(LL_n), axis=1).reshape(LL.shape[0], 1))
    

    def _entropy(self, distribution):
        log_dist = np.log2(distribution, where=distribution!=0)
        log_dist[log_dist == -np.inf] = 0

        if len(distribution.shape) == 1 or distribution.shape == (self._K**2 - self._K, self._L.size):
            return - np.sum(distribution * log_dist)
        elif len(distribution.shape) == 3:
            return - np.squeeze(np.sum(distribution * log_dist, axis=2).sum(axis=1))
        else:
            return - np.sum(distribution * log_dist, axis=1)
    
    # Background methods for discrete world representation
    def _models_to_links(self, models_probs, intervention=None):
        s = self._K**2 - self._K
        links_probs = np.zeros((s, self._num_links))
        for j in range(s):
            for k in range(self._num_links):
                links_probs[j, k] = np.sum(models_probs[self._indexed_space[:,j] == k])

        if intervention:
            links_probs[self.causes_idx[intervention], :] = 0 # Not sure yet about value
            
        return links_probs


    def _links_to_models(self, links_probs):
        return links_probs[np.arange(links_probs.shape[0]), self._indexed_space].prod(axis=1)

    
    def _build_space(self, links, as_matrix=False):
        a = links 
        c = len(links)
        s = self._K**2 - self._K

        S = np.zeros((c**s, s))

        for i in range(s):
            ou = np.tile(a, (int(c**(s-i-1)), 1)).flatten('F')
            os = tuple(ou for _ in range(c**i))
            o = np.concatenate(os)

            S[:, i] = o.T
        
        if not as_matrix:
            return S
        else:
            S_mat = np.zeros((c**s, self._K, self._K))

            for i in range(c**s):
                S_mat[i, :, :] = self._causality_matrix(S[i, :], fill_diag=1)
            
            return S_mat


# Local computation discrete agent
class Local_computations_omniscient_DIS(Discrete_IS):
    def __init__(self, N, K, prior_params, links, dt, theta, sigma, sample_params=True):
        super().__init__(N, K, prior_params, links, dt, theta, sigma, self._update_rule, sample_params=sample_params)

        self._prior_params = np.log(prior_params)
        self._init_priors()

        # Special parameters for faster computations
        self._links_lc_updates = np.tile(links.reshape((links.size, 1)), 3).T

        # Define own attractor mu, should be mu for each given the other two
        self._mus = self._attractor_mu(np.zeros(self._K))
        self._mus_history = [None for i in range(self._N)]

        

    def _update_rule(self, sensory_state, intervention=None):
        obs = sensory_state.s

        # Logic for updating
        log_likelihood_per_link = np.zeros(self._prior_params.shape)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    # Likelihood of observed the new values given the previous values for each model
                    log_likelihood = stats.norm.logpdf(obs[j], loc=self._mus[idx, :], scale=self._sigma*np.sqrt(self._dt))
                    # Normalisation step
                    likelihood_log = log_likelihood - np.amax(log_likelihood)
                    likelihood_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum()

                    ## If intervention, the probability of observing the new values is set to 1
                    if isinstance(intervention, tuple):
                        if j == intervention[0]:
                            likelihood_norm[:] = 1

                    log_likelihood_per_link[idx, :] = np.log(likelihood_norm)
                    idx += 1
        
        # Posterior params is the log likelihood of each model given the data
        log_posterior = self._posterior_params + log_likelihood_per_link

        # update mus
        self._update_mus(obs)

        return log_posterior

    
    # Background methods
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


# Local computation discrete agent
class Local_computations_interfocus_DIS(Discrete_IS):
    def __init__(self, N, K, prior_params, links, dt, theta, sigma, sample_params=True):
        super().__init__(N, K, prior_params, links, dt, theta, sigma, self._update_rule, sample_params=sample_params)

        self._prior_params = np.log(prior_params)
        self._init_priors()

        # Special parameters for faster computations
        self._links_lc_updates = np.tile(links.reshape((links.size, 1)), 3).T

        # Define own attractor mu, should be mu for each given the other two
        self._mus = self._attractor_mu(np.zeros(self._K))
        self._mus_history = [None for i in range(self._N)]

        

    def _update_rule(self, sensory_state, intervention=None):
        if not isinstance(intervention, tuple):
            return np.zeros(self._posterior_params.shape)

        obs = sensory_state.s

        # Logic for updating
        log_likelihood_per_link = np.zeros(self._prior_params.shape)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    if intervention[0] == i:
                        # Likelihood of observed the new values given the previous values for each model
                        log_likelihood = stats.norm.logpdf(obs[j], loc=self._mus[idx, :], scale=self._sigma*np.sqrt(self._dt))
                        # Normalisation step
                        likelihood_log = log_likelihood - np.amax(log_likelihood)
                        likelihood_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum()

                        log_likelihood_per_link[idx, :] = np.log(likelihood_norm)
                    else:
                        log_likelihood_per_link[idx, :] = np.zeros(self._L.size)
                    idx += 1
        
        # Posterior params is the log likelihood of each model given the data
        log_posterior = self._posterior_params + log_likelihood_per_link

        # update mus
        self._update_mus(obs)

        return log_posterior

    
    # Background methods
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

# Normative discrete agent
class Normative_DIS(Discrete_IS):
    def __init__(self, N, K, prior_params, links, dt, theta, sigma, sample_params=True):
        super().__init__(N, K, prior_params, links, dt, theta, sigma, self._update_rule, sample_params=sample_params)

        # Transfrom priors from links to graph representation
        prior = self._links_to_models(prior_params)
        self._prior_params = np.log(prior)
        self._init_priors()

        self._mus = self._attractor_mu(np.zeros(self._K))

        self._mus_history = [None for i in range(self._N)]

        
    # Update rule
    def _update_rule(self, sensory_state, intervention=None):
        obs = sensory_state.s

        # Likelihood of observed the new values given the previous values for each model
        likelihood_per_var = stats.norm.logpdf(obs, loc=self._mus, scale=self._sigma*np.sqrt(self._dt)) # Compute probabilities

        # Normalisation step
        likelihood_log = likelihood_per_var - np.amax(likelihood_per_var, axis=0)
        likelihood_per_var_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum(axis=0)
        
        ## If intervention, the probability of observing the new values is set to 1
        if isinstance(intervention, tuple):
            likelihood_per_var_norm[:, intervention[0]] = 1
 
        # Compute and normalise probabilities of each model given the previous and new values
        likelihood_to_prop = likelihood_per_var_norm.prod(axis=1)
        likelihood_over_models = likelihood_to_prop / likelihood_to_prop.sum()
        
        # Posterior params is the log likelihood of each model given the data
        ## The where argument is a problem, it makes it so models that are so unlikely that their probability is essentially 0 don't have their log likelihood penalised
        ## Cannot achieve numerical stability without it
        log_posterior = self._posterior_params + np.log(likelihood_over_models, where=likelihood_over_models!=0)
        #log_posterior = self._posterior_params + np.log(likelihood_over_models)

        # Update mus
        self._update_mus(obs)

        # Return log_posterior over models
        return log_posterior


    # Background methods
    def _update_mus(self, obs):
        self._mus_history[self._n] = self._mus
        self._mus = self._attractor_mu(obs)


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



    
    

    








