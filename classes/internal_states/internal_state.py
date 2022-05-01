from math import log
import numpy as np
from numpy.random.mtrand import sample
import pandas as pd
from scipy import stats
from scipy.stats import distributions

from classes.action_states.experience_discrete_3D_AS import Experience_discrete_3D_AS



# Main Internal state class
class Internal_state():
    def __init__(self, N, K, update_func, update_func_args=[], prior_param=None):
        self._N = N
        self._n = 0
        self._K = K

        self._prior_param = prior_param # Can either be a temperature parameter if the internal is discrete or a vector of standard deviation if it is continuous

        self._prior_params = None

        # p(i_t|s_t, i_t-1): must be a function of sensory states and the last action
        self._p_i_g_s_i = update_func
        self._p_i_g_s_i_args = update_func_args

        # Fitting parameters
        self._realised = False # True if data already exists
        self._fitting_judgement = False # True if fitting judgement data
        

    # General update function that all internal state object must satisfy
    ## Parameters can change but there must always be current set of posterior parameters & a history of the parameters at each time step
    def update(self, sensory_state, action_state):

        self._posterior_params_history[self._n] = self._posterior_params

        self._posterior_params = self._p_i_g_s_i(sensory_state, action_state, *self._p_i_g_s_i_args)
        
        self._n += 1
        
        
        if self._realised:
            if self._fitting_judgement:
                # Update history
                self._log_likelihood_history[self._n] = self._log_likelihood

                judgement_log_prob = 0
                j_data = self._judgement_data[self._n-1, :]

                if np.sum(np.isnan(j_data) != True) > 0:
                    link_idx = np.argmax(np.isnan(j_data) != True)
                    link_value = j_data[link_idx]

                    judgement_log_prob = self.posterior_PF_link(link_idx, link_value, log=True)

                    self._judgement_current[link_idx] = link_value

                elif self._n == self._N:
                    # Fit final judgement
                    judge_diff = self._judgement_current != self._judgement_final


                    for i in range(judge_diff.size):
                        if judge_diff[i]:
                            link_idx = i
                            link_value = self._judgement_final[link_idx]

                            judgement_log_prob += self.posterior_PF_link(link_idx, link_value, log=True)

                            self._judgement_current[link_idx] = link_value

                    print('Fiiting final judgement:', judgement_log_prob)

                self._log_likelihood += judgement_log_prob
            else:
                judgement_log_prob = 0
                self._log_likelihood += judgement_log_prob
        
            return judgement_log_prob
        

    def load_judgement_data(self, judgement_data, final_judgement, fit_judgement=True):
        self._judgement_data = judgement_data
        self._judgement_final = final_judgement

        self._judgement_current = np.empty(self._K**2 - self._K)
        self._judgement_current[:] = np.nan

        self._log_likelihood = 0
        self._log_likelihood_history = np.zeros(self._N+1)

        self._realised = True
        self._fitting_judgement = fit_judgement

    
    # Roll back internal state by a given number of step
    ## Used mostly for action selection
    def rollback(self, back=np.Inf):
        if back > self._N or back > self._n:
            self._n = 0
            self._reset_priors()
        else:
            self._n -= back

            self._posterior_params = self._posterior_params_history[self._n]
            # Reset Action values, seq and planned action from n to N
            for n in range(self._n+1, self._N):
                self._posterior_params_history[n] = None
                

    # Utility functions
    def initialise_prior_distribution(self, prior_judgement=None):
        self._prior_params = self._generate_prior_from_judgement(prior_judgement, self._prior_param) # Depends on continuous or discrete IS
        self._local_prior_init() # Model specific transformations of the prior
        self._posterior_params = self._prior_params
        self._posterior_params_history = [None for i in range(self._N)]
        self._posterior_params_history[0] = self._prior_params

        # Compute prior entropy
        self._prior_entropy = self.posterior_entropy
        self._prior_entropy_over_link = self.posterior_entropy_over_links

    
    def _reset_prior(self):
        # Simply reset priors to intial states
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
    def __init__(self, N, K, links, dt, update_func, update_func_args=[], generate_sample_space=True, sample_params=True, prior_param=None, smoothing=0):

        super().__init__(N, K, update_func, update_func_args=update_func_args, prior_param=prior_param)

        self._num_links = len(links)
        self._dt = dt
        self._smoothing_temp = smoothing
        
        # Sample parameter estimates
        if sample_params:
            # Sample key variables according to Davis, Rehder, Bramley (2018)
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
            self._L = links

        # Build representational spaces
        ## if generate sample space == True, build sample space, else, wait for call of the add_sample_space method call
        if generate_sample_space:
            self._sample_space = self._build_space(links) # Sample space as set of vectors with link values
            self._indexed_space = self._build_space(np.arange(len(links))).astype(int) # Same as above but link values are indices
            self._sample_space_as_mat = self._build_space(links, as_matrix=True) # Sample space as a set of causality matrices, i.e. one for each model
        else:
            self._sample_space = None
            self._indexed_space = None
            self._sample_space_as_mat = None
        

    # Properties
    @property
    def posterior(self):
        posterior = self._likelihood(self._posterior_params)
        smoothed_posterior = self._smooth_softmax(posterior)
        return smoothed_posterior

    @property
    def posterior_unsmoothed(self):
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
    def MAP(self):
        graph_idx = np.argmax(self.posterior_over_models)
        return self._sample_space[graph_idx]

    @property
    def posterior_entropy(self):
        return self._entropy(self.posterior_over_models)

    @property
    def posterior_entropy_unsmoothed(self):
        if len(self.posterior_unsmoothed.shape) == 1:
            return self._entropy(self.posterior_unsmoothed)
        else:
            return self._entropy(self._links_to_models(self.posterior_unsmoothed))

    @property
    def prior_entropy(self):
        return self._prior_entropy

    @property
    def prior_entropy_over_links(self):
        return self._prior_entropy_links

    @property
    def posterior_entropy_over_links(self):
        return self._entropy(self.posterior_over_links)
    
    @property
    def entropy_history(self):
        posterior_history = self._likelihood(self._posterior_params_history)
        return self._entropy(posterior_history)

    # Return a posterior over model for the given index between 0 and N
    def posterior_over_models_byidx(self, idx):
        posterior = self._likelihood(self._posterior_params_history[idx])
        if len(self.posterior.shape) == 1:
            return self._smooth(posterior)
        else:
            return self._links_to_models(self._smooth(posterior))

    # Samples the posterior, the number of samples is given by the size parameter
    def posterior_sample(self, size=1, uniform=False, as_matrix=False):
        if uniform:
            graph_idx = np.random.choice(np.arange(self._sample_space.shape[0]), size=size)
        else:
            graph_idx = np.random.choice(np.arange(self._sample_space.shape[0]), size=size, p=self.posterior_over_models)
            
        if as_matrix:
            return self._sample_space_as_mat[graph_idx].squeeze()
        else:
            return self._sample_space[graph_idx].squeeze()

    # PMF of the posterior for a given graph
    def posterior_PF(self, graph, log=False):
        if not log:
            return self.posterior_over_models[np.where((self._sample_space == graph).all(axis=1))[0]]
        else:  
            return np.log(self.posterior_over_models[np.where((self._sample_space == graph).all(axis=1))[0]])

    # PMF of the posterior for a given link
    def posterior_PF_link(self, link_idx, link_value, log=False):
        value_idx = np.squeeze(np.where(self._L == link_value)[0])
        if not log:
            prob = self.posterior_over_links[link_idx, value_idx]
            return prob
        else:
            log_prob = np.log(self.posterior_over_links[link_idx, value_idx])
            return log_prob


    # Prior initialisation
    def _generate_prior_from_judgement(self, prior_judgement, temperature):

        if type(prior_judgement) == np.ndarray:
            prior_j = prior_judgement  
            temp = temperature  
        else:
            prior_j = np.zeros(self._K**2 - self._K)
            temp = 0

        distances = ((self._sample_space - prior_j)**2).sum(axis=1)**(1/2)
        norm_distances = 1 - distances / distances.max()
    
        softmax_prior = self._softmax(norm_distances, temp)

        return self._models_to_links(softmax_prior)

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
    

    def _smooth(self, dist):
        if self._smoothing_temp == None:
            return dist
        
        if len(dist.shape) == 1:
            dist = dist.reshape((1, dist.size))

        indices = np.tile(np.arange(dist.shape[1]), (dist.shape[0], 1))
        max_dist = np.argmax(dist, axis=1).reshape((dist.shape[0], 1))

        smoother = ( dist.shape[1] - np.abs(indices - max_dist) ) / dist.shape[1]

        certainty_coef = np.exp(- self._entropy(dist, custom=True)) * self._smoothing_temp

        smoothed_values = dist + certainty_coef * smoother

        return smoothed_values / smoothed_values.sum(axis=1).reshape((dist.shape[0], 1))

    def _smooth_softmax(self, dist):
        if self._smoothing_temp == None:
            return dist
        
        if len(dist.shape) == 1:
            dist_n = dist.reshape((1, dist.size))
        else:
            dist_n = dist
            
        exp_dist = np.exp(dist_n * self._smoothing_temp)
        norm = exp_dist.sum(axis=1).reshape((exp_dist.shape[0], 1))

        new_dist = (exp_dist / norm).reshape(dist.shape)
        return new_dist


    def _entropy(self, distribution, custom=False):
        log_dist = np.log2(distribution, where=distribution!=0)
        log_dist[log_dist == -np.inf] = 0

        if len(distribution.shape) == 1 or custom:
            return - np.sum(distribution * log_dist)
        elif len(distribution.shape) == 3:
            return - np.squeeze(np.sum(distribution * log_dist, axis=2).sum(axis=1))
        else:
            return - np.sum(distribution * log_dist, axis=1)


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

    
    # Sample space related methods
    def add_sample_space_env(self, triple_of_spaces):
        # Add sample space manually
        self._sample_space, self._indexed_space, self._sample_space_as_mat = triple_of_spaces


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

    
    def _softmax(self, d, temp=1):
        if len(d.shape) == 1:
            return np.exp(d*temp) / np.exp(d*temp).sum()
        else:
            return np.exp(d*temp) / np.exp(d*temp).sum(axis=1, keepdims=1)




# Internal state using a continuous probability distribution to represent the external states
## Need to be able to express discrete probability values
class Continuous_IS(Internal_state):
    def __init__(self, N, K, links, dt, update_func, update_func_args=[], generate_sample_space=True, sample_params=True, prior_param=None , smoothing=0):
        super().__init__(N, K, update_func, update_func_args=update_func_args, prior_param=prior_param)

        # Smoothing temperature
        self._smoothing_temp = smoothing

        # General dt
        self._dt = dt

        # Sample parameter estimates
        if sample_params:
            # Sample key variables according to Davis, Rehder, Bramley (2018)
            self._L = np.zeros(len(links))
            for i, l in enumerate(links):
                if l < 0:
                    self._L[i] = -1 * stats.gamma.rvs(100*np.abs(l), scale=1/100)
                elif l == 0:
                    self._L[i] = 0
                else:
                    self._L[i] = stats.gamma.rvs(100*l, scale=1/100)

            self._interval = np.abs(self._L[0] - self._L[1]) / 2
            
        else:
            # Assume perfect knowledge
            self._L = links
            self._interval = np.abs(self._L[0] - self._L[1]) / 2
        
        
        # Build representational spaces
        ## if generate sample space == True, build sample space, else, wait for call of the add_sample_space method call
        if generate_sample_space:
            self._sample_space = self._build_space(links) # Sample space as set of vectors with link values
            self._indexed_space = self._build_space(np.arange(len(links))).astype(int) # Same as above but link values are indices
            self._sample_space_as_mat = self._build_space(links, as_matrix=True) # Sample space as a set of causality matrices, i.e. one for each model
        else:
            self._sample_space = None
            self._indexed_space = None
            self._sample_space_as_mat = None

    # Properties
    @property
    def posterior_params(self):
        return self._posterior_params

    @property
    def posterior(self):
        return self._smooth(self._posterior_pmf(self._posterior_params))
    
    @property
    def posterior_over_links(self):
        return self.posterior

    @property
    def posterior_over_models(self):
        return self._links_to_models(self.posterior) 

    @property
    def MAP_continuous(self):
        return self._argmax()

    @property
    def MAP(self):
        graph_idx = np.argmax(self.posterior_over_models)
        return self._sample_space[graph_idx]

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
    def prior_entropy(self):
        return self._prior_entropy

    @property
    def prior_entropy_over_links(self):
        return self._prior_entropy_links
    
    @property
    def posterior_entropy(self):
        return self._entropy(self.posterior_over_links, custom=True)

    @property
    def posterior_entropy_over_links(self):
        return self._entropy(self.posterior_over_links)
    
    @property
    def entropy_history(self):
        entropy_history = np.zeros(len(self._posterior_params_history))
        for i in range(entropy_history.size):
            entropy_history[i] = np.sum(self._entropy(self._posterior_pmf(self._posterior_params_history[i])))  
        return entropy_history
    
    @property
    def posterior_differential_entropy(self):
        return np.sum(self._diff_entropy(self.posterior_params))

    @property
    def posterior_differential_entropy_over_links(self):
        return self._diff_entropy(self.posterior_params)

    @property
    def differential_entropy_history(self):
        entropy_history = np.zeros(len(self._posterior_params_history))
        for i in range(entropy_history.size):
            entropy_history[i] = np.sum(self._diff_entropy(self._posterior_params_history[i]))
        return entropy_history


    # Return a posterior over model for the given index between 0 and N
    def posterior_over_models_byidx(self, idx):
        if len(self.posterior.shape) == 1:
            return self._smooth(self._posterior_pmf(self._posterior_params_history[idx]))
        else:
            return self._links_to_models(self._smooth(self._posterior_pmf(self._posterior_params_history[idx])))


    # Samples the posterior, the number of samples is given by the size parameter
    def posterior_sample(self, size=1, as_matrix=False):
        # Needs to be specifically defined per sub class
        sample = self._sample_distribution(size=size)
            
        if as_matrix:
            if size == 1:
                return self._causality_matrix(sample)
            else:
                sample_matrix = np.zeros((size, self._K, self._K))
                for i in range(size):
                    sample[i, :, :] = self._causality_matrix(sample[i, :])
                return sample_matrix
        else:
            return sample
            

    # PMF of the posterior for a given graph
    def posterior_PF(self, graph, log=False):
        if log:
            log_posterior = np.log(self._pdf(graph))
            return np.sum(log_posterior)
        else:
            return np.prod(self._pdf(graph))


    # PMF of the posterior for a given link
    def posterior_PF_link(self, link_idx, link_value, log=False):
        if log:
            return np.log(self._link_pdf(link_idx, link_value))
        else:
            return self._link_pdf(link_idx, link_value)

        
    # Prior initialisation
    def _generate_prior_from_judgement(self, prior_judgement, sigma):
        if prior_judgement:
            means = prior_judgement
            s = sigma
        else:
            means = np.zeros(self._K**2 - self._K)
            s = 1e10

        if type(s) == np.ndarray:
            sd = s
        else:
            sd = s * np.ones(means.shape)

        return np.array([means, sd]).T
    

    def _diff_entropy(self, parameters):
        # Needs to be specifically defined per sub class
        return self._entropy_distribution(parameters)

    
    def _entropy(self, distribution, custom=False):
        log_dist = np.log2(distribution, where=distribution!=0)
        log_dist[log_dist == -np.inf] = 0

        if len(distribution.shape) == 1 or custom:
            return - np.sum(distribution * log_dist)
        elif len(distribution.shape) == 3:
            return - np.squeeze(np.sum(distribution * log_dist, axis=2).sum(axis=1))
        else:
            return - np.sum(distribution * log_dist, axis=1)

    
    def _smooth(self, dist):
        if self._smoothing_temp == None:
            return dist
        
        if len(dist.shape) == 1:
            dist = dist.reshape((1, dist.size))

        indices = np.tile(np.arange(dist.shape[1]), (dist.shape[0], 1))
        max_dist = np.argmax(dist, axis=1).reshape((dist.shape[0], 1))

        smoother = ( dist.shape[1] - np.abs(indices - max_dist) ) / dist.shape[1]

        certainty_coef = np.exp(- self._entropy(dist, custom=True)) * self._smoothing_temp

        smoothed_values = dist + certainty_coef * smoother

        return smoothed_values / smoothed_values.sum(axis=1).reshape((dist.shape[0], 1))


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

    
    # Sample space related methods
    def add_sample_space_env(self, triple_of_spaces):
        # Add sample space manually
        self._sample_space, self._indexed_space, self._sample_space_as_mat = triple_of_spaces

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

    def softmax(self, d, temp=1):
        return np.exp(d/temp) / np.exp(d/temp).sum(axis=1).reshape((d.shape[0], 1))
        



    
    

    








