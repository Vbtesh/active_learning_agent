from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np

# Local computation discrete agent
class Local_computations_interfocus_DIS(Discrete_IS):
    def __init__(self, N, K, links, dt, theta, sigma, decay_type, decay_rate=0.65, generate_sample_space=True, sample_params=False, prior_param=None, smoothing=False):
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
        
        self._prior_param = prior_param

        # Special parameters for faster computations
        self._links_lc_updates = np.tile(links.reshape((links.size, 1)), 3).T

        # Define own attractor mu, should be mu for each given the other two
        self._mus = self._attractor_mu(np.zeros(self._K))
        self._mus_history = [None for i in range(self._N)]

        if decay_type == 'exponential':
            self._power_update_coef = self._exponential_decay
        elif decay_type == 'sigmoid':
            self._power_update_coef = self._sigmoid_decay

        self._decay_rate = decay_rate if decay_rate > 1e-1 else 1e-1

        self._last_action = None
        self._last_action_len = None
        self._last_instant_action = None
        self._last_real_action = None
        self._last_action_idx = 0
        self._last_action_end = None

        self._power_update_coef_history = np.zeros(self._N)


    def _update_rule(self, sensory_state, action_state):
        intervention = action_state.a
        
        obs = sensory_state.s
        
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
                self._last_action_end = None

                self._last_action = action_state.a_real

        
            elif self._last_action and action_state.a_real:
                if not self._last_instant_action:
                    # CHANGE OF ACTION
                    # Action length is
                    self._last_action_len = action_state.a_len_real      
                    # Reset last action index
                    self._last_action_idx = 0
                    self._last_action_end = None

                    self._last_action = action_state.a_real

            elif self._last_instant_action and not action_state.a_real:
                # Stopped acting
                self._last_action_end = 1               
    
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
                self._last_action_end = None

                self._last_action = intervention
                self._last_instant_action = intervention
            elif self._last_action and intervention:
                if not self._last_instant_action:
                    # Action length is
                    self._last_action_len = action_state.a_len     
                    # Reset last action index
                    self._last_action_idx = 0
                    self._last_action_end = None

                    self._last_action = intervention
                    self._last_instant_action = intervention

            elif self._last_instant_action and not intervention:
                # Stopped acting
                self._last_action_end = 1  

    
        # Logic for updating
        # Compute power update coefficient

        delay = self._last_action_len
        power_coef = self._power_update_coef(delay)
        self._power_update_coef_history[self._n] = power_coef

        log_likelihood_per_link = np.zeros(self._posterior_params.shape)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    if j == self._last_action[0] or i != self._last_action[0]:
                        log_likelihood_per_link[idx, :] = np.zeros(self._L.size)
                        
                    else:
                        # Likelihood of observed the new values given the previous values for each model
                        log_likelihood = power_coef * stats.norm.logpdf(obs[j], loc=self._mus[idx, :], scale=self._sigma*np.sqrt(self._dt))
                        # Normalisation step
                        likelihood_log = log_likelihood - np.amax(log_likelihood)
                        likelihood_norm = np.exp(likelihood_log) / np.exp(likelihood_log).sum()

                        log_likelihood_per_link[idx, :] = np.log(likelihood_norm)
                    idx += 1
        
        # Posterior params is the log likelihood of each model given the data
        log_posterior = self._posterior_params + log_likelihood_per_link

        # update mus
        self._update_mus(obs)

        self._last_obs = obs
        self._last_action_idx += 1

        if action_state.realised:
            if not self._last_instant_action and not action_state.a_real:
                self._last_action_end += 1
            self._last_instant_action = action_state.a_real
        else:
            if not self._last_instant_action and not intervention:
                self._last_action_end += 1
            self._last_instant_action = intervention

        return log_posterior


    # Background methods
    ## Prior initialisation specific to model:
    def _local_prior_init(self):
        self._prior_params = np.log(self._prior_params)

    # Power update coefficients
    ## Exponential decay
    def _exponential_decay(self, delay):
        if not self._last_action_end:
            return 1
        else:
            return self._decay_rate**self._last_action_end

    ## Sigmoid decay
    def _sigmoid_decay(self, delay):
        return 1 / (1 + self._decay_rate**(- self._last_action_idx + delay))

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