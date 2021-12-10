from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np

# Local computation discrete agent
class Change_based_DIS(Discrete_IS):
    def __init__(self, N, K, prior_params, links, dt, theta, sigma, sample_params=True, smoothing=False):
        super().__init__(N, K, prior_params, links, self._update_rule, sample_params=sample_params, smoothing=smoothing)

        prob_prior_params = self._values_to_probs(prior_params)
        self._prior_params = prob_prior_params
        self._init_priors()

        # Special parameters for faster computations
        self._links_lc_updates = np.tile(links.reshape((links.size, 1)), 3).T


    def _update_rule(self, sensory_state, intervention=None):
        obs = sensory_state.s

        # Logic for updating
        ## Get change
        ## Update evidence or values (posterior value)
        ## Transform into prob
        log_posterior = 0
        return log_posterior

    
    # Background methods
    def _values_to_probs(self, values):
        # Normalise OR softmax values to probabilities
        pass

