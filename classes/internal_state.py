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

        # Maintain a log likelihood for fitting
        self._log_likelihood = 0
        self._log_likelihood_history = np.zeros(self._N+1)

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

            j_data = self._judgement_data[self._n-1, :]
            if np.sum(np.isnan(j_data) != True) > 0:
                link_idx = np.argmax(np.isnan(j_data) != True)
                link_value = j_data[link_idx]

                judgement_log_prob = self.posterior_PMF_link(link_idx, link_value, log=True)

                self._log_likelihood += judgement_log_prob
                return judgement_log_prob
            else:
                return 0
        

    def load_judgement_data(self, judgement_data, final_judgement):
        self._judgement_data = judgement_data
        self._final_judgement = final_judgement

        self._log_likelihood = 0
        self._log_likelihood_history = np.zeros(self._N + 1)

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
            for n in range(self._n+1, self._N+1):
                self._posterior_params_history[n] = None
                

    # Utility functions
    def _init_priors(self):
        self._posterior_params = self._prior_params
        self._posterior_params_history = [None for i in range(self._N+1)]
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

    
    def _causality_vector(self, link_mat):
        s = link_mat.shape[0]**2 - link_mat.shape[0]

        causal_vec = np.zeros(s)

        idx = 0
        for i in range(link_mat.shape[0]):
            for j in range(link_mat.shape[0]):
                if i != j:
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
        return self._models_to_links(self.posterior)

    @property
    def map(self):
        graph_idx = np.argmax(self.posterior)
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
            return self.posterior[np.where((self._sample_space == graph).all(axis=1))[0]]
        else:  
            return np.log(self.posterior[np.where((self._sample_space == graph).all(axis=1))[0]])

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
        else:
            LL = log_likelihood
        if len(LL.shape) == 1:
            LL = LL.reshape(1, LL.shape[0])
        LL_n = LL - np.amax(LL, axis=1).reshape(LL.shape[0], 1)
        return np.squeeze(np.exp(LL_n) / np.sum(np.exp(LL_n), axis=1).reshape(LL.shape[0], 1))
    

    def _entropy(self, distribution):
        log_dist = np.log2(distribution, where=distribution!=0)

        if len(distribution.shape) == 1:
            return - np.sum(distribution * log_dist)
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

    
# Normative agent
class Normative_DIS(Discrete_IS):
    def __init__(self, N, K, prior_params, prior_sample_size, links, dt, theta, sigma, sample_params=True):
        super().__init__(N, K, prior_params, links, dt, theta, sigma, self._update_rule, sample_params=sample_params)

        # Transfrom priors from links to graph representation
        prior = self._links_to_models(prior_params)
        self._prior_params = np.log(prior) * prior_sample_size
        self._init_priors()

        self._mus = self._attractor_mu(np.zeros(self._K))

        self._mus_history = [None for i in range(self._N)]

        
    # Update rule
    def _update_rule(self, sensory_state, intervention=None):
        obs = sensory_state.s

        # Likelihood of observed the new values given the previous values for each model
        likelihood_per_var = stats.norm.logpdf(obs, loc=self._mus, scale=np.sqrt(self._dt)) # Compute probabilities

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
        log_posterior = self._posterior_params + np.log(likelihood_over_models, where=likelihood_over_models!=0)

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



    
    

    










### OLD INTERNAL STATE CLASS (MODEL CLASS)

class Internal_state_old():
    def __init__(self, N, K, links, theta, dt, sigma, link_prior, prior_sample_size, likelihood='LL', dist_rule='softmax', sample_params=True):
        
        # Initialise parameters
        self._N = N
        self._n = 0
        self._K = K
        self._dt = dt
        self._num_links = len(links)
    
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
        
        # Likelihood function for output
        self._likelihood_type = likelihood
        # Distribution rule for weights
        self._dist_rule = dist_rule

        # Build representational spaces
        self._sample_space = self._build_space(links) # Sample space as set of vectors with link values
        self._indexed_space = self._build_space(np.arange(len(links))).astype(int) # Same as above but link values are indices
        self._sample_space_as_mat = self._build_space(links, as_matrix=True) # Sample space as a set of causality matrices, i.e. one for each model

        self.causes_idx = [self._map_cause_to_effect(i, K**2-K) for i in range(K)] # Internal variables for converting from distribution over models to distribution over links


        # Initialise probability distributions
        self._raw_link_prior = link_prior
        self._raw_model_prior = self._links_to_models(link_prior)
        self._prior_sample_size = prior_sample_size
        ## Links
        ### Weights
        self._link_prior_weights = self._raw_link_prior * self._prior_sample_size    # Fixed prior weight matrix over links
        self._link_weights = self._link_prior_weights                       # Updated weight matrix over links
        ### Probabilities
        self._link_prior = self._weights_to_probs(self._link_prior_weights) # Fixed prior probability matrix over links
        self._link_probs = self._link_prior                                 # Updated probability matrix over links

        ## Models
        ### Weights
        self._model_prior_weights = self._raw_model_prior * self._prior_sample_size # Fixed prior weight matrix over models
        self._model_weights = self._model_prior_weights                                          # Updated weight matrix over links
        ### Probability distribution over models
        self._model_prior = self._weights_to_probs(self._model_weights) 
        self._model_probs = self._model_prior  

        ### Log likelihood of each possible model (can be normalised for formal posterior)
        self._model_LL_prior = np.log(self._raw_model_prior, where=self._raw_model_prior!=0) * self._prior_sample_size         
        self._model_LL = self._model_LL_prior


        # Set up a history list for all posteriors over graphs
        self._history_weights = np.zeros((self._N+1, self._model_weights.size))
        self._history_weights[0, :] = self._model_weights
        self._history_LL = np.zeros((self._N+1, self._model_probs.size))
        self._history_LL[0, :] = self._model_LL

    
    def update(self, x_prev, x, intervention=None):
        # Compute one step transition probabilities over models
        ## Attractor for each models
        mu = x_prev + (self._attractor_mu(x_prev) - x_prev) * self._theta * self._dt

        ## Likelihood of observed the new values given the previous values for each model
        posterior_per_var = stats.norm.logpdf(x, loc=mu, scale=np.sqrt(self._dt)) # Compute probabilities

        ## Normalisation step
        posterior_log = posterior_per_var - np.amax(posterior_per_var, axis=0)
        posterior_per_var_norm = np.exp(posterior_log) / np.exp(posterior_log).sum(axis=0)
        
        ## If intervention, the probability of observing the new values is set to 1
        if type(intervention) == np.ndarray:
            posterior_per_var_norm[:, intervention[0]] = 1
 
        ## Compute and normalise probabilities of each model given the previous and new values
        post_over_models = posterior_per_var_norm.prod(axis=1)
        post_over_models = post_over_models / post_over_models.sum()

        # Update posteriors
        self._model_weights += post_over_models
        self._model_LL += np.log(post_over_models, where=post_over_models!=0)

        # Record history
        self._history_weights[self._n+1] = self._model_weights
        #print('before update:', self.entropy_history)
        self._history_LL[self._n+1] = self._model_LL
        #print('after update:', self.entropy_history)

        # Update current sample size
        self._n += 1


    def rollback(self, back=np.Inf):
        if back > self._N or back > self._n:
            self._n = 0
        else:
            self._n -= back

        # Roll back posteriors
        self._model_weights = np.array(self._history_weights[self._n])
        self._history_weights[self._n+1:, :] = 0
        self._model_LL = np.array(self._history_LL[self._n])
        self._history_LL[self._n+1:, :] = 0

    
    # Properties
    @property
    def mle(self):
        if self._likelihood_type == 'LL':
            return self._sample_space[np.argmax(self._model_LL), :]
        else:
            return self._sample_space[np.argmax(self._model_weights), :]

    @property
    def posterior_models(self):
        if self._likelihood_type == 'LL':
            return self._likelihood(self._model_LL)
        else:
            return self._weights_to_probs(self._model_probs)


    @property
    def posterior_links(self):
        if self._likelihood_type == 'LL':
            LH = self._likelihood(self._model_LL)
        else:
            LH = self._weights_to_probs(self._model_probs)
        return self._models_to_links(LH)

    @property
    def posterior_history(self):
        if self._likelihood_type == 'LL':
            return self._likelihood(self._history_LL)
        else:
            return self._weights_to_probs(self._history_weights)

    @property
    def posterior_entropy(self):
        if self._likelihood_type == 'LL':
            LH = self._likelihood(self._model_LL)
        else:
            LH = self._weights_to_probs(self._model_probs)
        return self._entropy(LH)
    
    @property
    def entropy_history(self):
        if self._likelihood_type == 'LL':
            LH = self._likelihood(self._history_LL)
        else:
            LH = self._weights_to_probs(self._history_weights)
        return self._entropy(LH)
    
    @property
    def sample_space(self):
        return self._sample_space


    # Reports
    def prob_model(self, model):
        LH = self._likelihood(self._model_LL)
        return LH[np.where((self._sample_space == model).all(axis=1))[0]]

    
    def rank_models(self, num=10, model=None):
        df = pd.DataFrame()
        df['probs'] = self._likelihood(self._model_LL)

        idx = df.sort_values('probs', axis=0, ascending=False).index 

        if type(model) == np.ndarray:
            return list(idx).index(np.where((self._sample_space == model).all(axis=1))[0]) + 1
        else:
            return self._sample_space[idx[0:num+1]]


    # Background methods
    def _attractor_mu(self, x_prev): 
        mu = x_prev * (1 - np.abs(x_prev / 100)) + x_prev @ self._sample_space_as_mat
        return np.squeeze(mu)


    def _likelihood(self, LL):
        if len(LL.shape) == 1:
            LL = LL.reshape(1, LL.shape[0])
        LL_n = LL - np.amax(LL, axis=1).reshape(LL.shape[0], 1)
        return np.squeeze(np.exp(LL_n) / np.sum(np.exp(LL_n), axis=1).reshape(LL.shape[0], 1))


    def _weights_to_probs(self, weights, dist=None):
        if not dist:
            dist = self._dist_rule

        weights = weights.T

        if dist == 'weights':
            probs = weights / np.sum(weights, axis=0)
        elif dist == 'softmax':
            probs = np.exp(weights) / np.sum(np.exp(weights), axis=0)
        elif dist == 'squared_weights':
            probs = weights**2 / np.sum(weights**2, axis=0)

        return probs.T

    
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

    
    def _causality_vector(self, link_mat):
        s = link_mat.shape[0]**2 - link_mat.shape[0]

        causal_vec = np.zeros(s)

        idx = 0
        for i in range(link_mat.shape[0]):
            for j in range(link_mat.shape[0]):
                if i != j:
                    causal_vec[idx] = link_mat[i, j]
                    idx += 1

        return causal_vec


    def _map_cause_to_effect(self, var_idx, vec_size):
        vec_range = np.arange(vec_size)
        causal_mat = self._causality_matrix(vec_range)
        return causal_mat[np.arange(causal_mat.shape[0]) != var_idx, var_idx].astype(int)


    def _entropy(self, distribution):
        log_dist = np.log2(distribution, where=distribution!=0)

        if len(distribution.shape) == 1:
            return - np.sum(distribution * log_dist)
        else:
            return - np.sum(distribution * log_dist, axis=1)