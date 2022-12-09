from classes.internal_states.internal_state import Variational_IS
from scipy import stats
import numpy as np


"""
Mean field variational Agent

Holds beliefs about all parameters. Such beliefs are represented via variational distributions
This agent performs mean-field approximation:

    Q(theta_1, theta_2, ..., theta_k) = Q(theta_1)Q(theta_2)...Q(theta_k)

Updates are chosen by the active states.

Variational distribution (Qs) can be of any form but in general it is much simpler to use categorical
distribution as an expectation over their values must be computed for each update.

Theta has a normal distribution
Sigma has a discrete distribution over three values {1dt, 3dt, 6dt}
Gammas has a discrete distribution over each of the link {-1, -1/2, 0, 1/2, 1}.
Dalston Kingsland, London E8 2FAhack
One needs to create an indexed representation of such Qs with:
- Distributions types (Gaussian, discrete): to know how to update
- Entropies: to select which distribution to update
- Rules of update for each (function?) that given data and others Qs can compute the updated parameters

Rules of updates are function that update parameters and are called if the index of the Qs in the boolean selector 
from active states is set to True 
- Need a generic update if all distributions are discrete
- Need a specific update if theta is Gaussian: should be faster as less combinatorially complex


TO DO:
Redefine the methods in the Variational IS object
Prior initialisation
OK Think about what to output for posterior and entropy? 
OK Maybe create special version that output all the factors and keep the standard ones consistent with the general framework
Move to action state!!
"""


class MeanField_VIS(Variational_IS):
    def __init__(self, N, K, dt, parameter_set, certainty_threshold, generate_sample_space=True, prior_param=None, smoothing=False):
        super().__init__(N, K, dt, self._update_rule, generate_sample_space=generate_sample_space, prior_param=prior_param, smoothing=smoothing)

        # Collect observations to recompute mus
        self._obs_history = [None for _ in range(self._N+1)]
        self._obs_history[0] = np.zeros(self._K)

        self._epsilon = certainty_threshold

        self._param_names_dict = parameter_set
        
        self._parameter_set = parameter_set
        names = [p_name for p_name in parameter_set.keys()]
        self._num_factors = len(names)
        self._param_names_list = np.array(names, dtype=object)
        self._param_index_list = np.arange(len(names))

        self._link_params_bool = np.ones(self._num_factors, dtype=bool)
        for i, name in enumerate(names):
            if self._parameter_set[name]['type'] == 'no_links':
                self._link_params_bool[i] = 0

        self._link_names_matrix = self._construct_link_matrix(K)

        self._param_subsets_dict = self._construct_parameter_subset_dict()

        # Initialise Qs
        ## Always in the log space
        self._prior_Qs = np.array([np.log(self._param_names_dict[name]['prior']) for name in names], dtype=object)
        self._Qs = self._prior_Qs



    # Update rule
    def _update_rule(self, sensory_state, action_state):
        """
        Action state should hold a specific parameter for the variational agent.
        A indicator vector stating which factor to update
        
        For each selected factor
            Use obsrved data and depend factor to update it
            If link value, use all non link factors and all other factors sharing the same effect
            Else, use all other non link factor and all factors have the current intervened variable as a cause
        """
        to_update = np.array([0, 1, 1, 0, 0, 0, 0, 0]).astype(bool)

        action_fromstate = action_state.a
        if not action_fromstate:
            return self._Qs
        else:
            action = action_fromstate[0]

        X = sensory_state.s
        X_prev = sensory_state.s_prev

        params_to_update = self._param_names_list[to_update]

        dependencies_str = [None for _ in params_to_update]
        dependencies_array = np.zeros(len(dependencies_str), to_update.size, dtype=bool)

        var_to_consider = np.zeros((params_to_update.size, self._K), dtype=bool)

        for i, param in enumerate(params_to_update):
    
            dependencies_str[i] = self._param_names_dict[param]['dependencies_as_str']
            dependencies_array[i, :] = self._param_names_dict[param]['dependencies_as_bool']

            if self._param_names_dict[param]['type'] == 'no_link':
                dep_bool = np.zeros(self._link_name_matrix.shape, dtype=bool)
                for i in range(self._K):
                    for j in range(self._K):
                        if i != j and j != action:
                            dep_bool[i, j] = 1
                additional_dependencies = self._link_names_matrix[dep_bool]
                add_dep_bool = np.in1d(self._param_names_list, additional_dependencies)
                dependencies_str[i] += additional_dependencies.tolist()
                dependencies_array[i, :][add_dep_bool] = 1

                var_to_consider[i, :] = np.ones(self._K, dtype=bool)
                var_to_consider[i, action] = 0
            else:
                var_to_consider[i, int(params_to_update[i][-1])] = 1

            # Now that dependencies needed to update have been found
            ## Generate a list of all parameter values and the associated joints
            dep_str = ','.join(dependencies_str[i])
            parameter_values = self._param_subsets_dict[dep_str]
            expectation_probabilities = self._construct_parameter_combinations(dependencies_str[i], self._Qs[dependencies_array[i, :]])

            # Perform the update
            # Compute the expectation
            if param in self._causal_link_names:
                expectation_quantities = self._causal_link_expectation(param, self._param_names_dict[param]['values'], action, X, X_prev, dependencies_array[i, :], parameter_values, var_to_consider)
            else:
                expectation_quantities = self._parameter_expectation(param, self._param_names_dict[param]['values'], action, X, X_prev, dependencies_array[i, :], parameter_values, var_to_consider)

            # Unnormalised probability of observing this transition
            data_log_probability_unnormalised = np.sum(expectation_probabilities * expectation_quantities, axis=1)

            # Update considered belief
            self._Qs[i] += data_log_probability_unnormalised
        
        return self._Qs

    """
    Update functions for causal link and other parameters
    """
    def _causal_link_expectation(self, belief_name, belief_values, action, X, X_prev, parameter_names, parameter_values, var_to_consider):
        
        out = np.zeros((belief_values.size, parameter_values.shape[0]))
        for k in range(self._K):
            if var_to_consider[k] == 0:
                continue
            
            current_var = np.zeros(self._K).astype(bool)
            current_var[k] = 1

            sigmas = parameter_values[:, parameter_names.index('sigma')]
            thetas = parameter_values[:, parameter_names.index('theta')]

            links = np.ones((belief_values.size, self._K, parameter_values.shape[0]))
            
            links[:, ~current_var, :] = np.tile(parameter_values[:, 2:], (belief_values.size, 1, 1)).reshape((belief_values.size, values_space[:, 2:].shape[1], values_space[:, 2:].shape[0]))
            bvs = np.tile(belief_values, (parameter_values[:, 2:].shape[0], 1)).T
            links[:, int(belief_name[0]), :] = bvs

            regularisor = -1 * X_prev[k] * (np.abs(X_prev[k]) / 100)

            mus = thetas * (X_prev @ links + regularisor - X_prev[k]) * self._dt

            out += stats.norm.logpdf(X[k], loc=mus, scale=sigmas)

        return out 


    def _parameter_expectation(self, belief_name, belief_values, action, X, X_prev, parameter_names, parameter_values, var_to_consider):
        
        out = np.zeros((belief_values.size, parameter_values.shape[0]))
        for k in range(self._K):
            if var_to_consider[k] == 0:
                continue
            
            current_var = np.zeros(self._K).astype(bool)
            current_var[k] = 1

            if belief_name == 'sigma':
                thetas = parameter_values[:, parameter_names.tolist().index('theta')]
                sigmas = belief_values
            else:
                sigmas = parameter_values[:, parameter_names.tolist().index('sigma')]
                thetas = belief_values

            links_indices = [parameter_names.tolist().index(param) for param in parameter_names[1:] if int(param[-1]) == k]
            #print(links_indices)
            links = np.ones((self._K, parameter_values.shape[0]))
            links[~current_var, :] = parameter_values[:, links_indices].T

            regularisor = -1 * X_prev[k] * (np.abs(X_prev[k]) / 100)

            if belief_name == 'theta':
                mus = thetas.reshape((thetas.size, 1)) * (X_prev @ links + regularisor - X_prev[k]) * self._dt
                out += stats.norm.logpdf(X[k], loc=mus, scale=sigmas)
            else:
                mus = thetas * (X_prev @ links + regularisor - X_prev[k]) * self._dt
                out += stats.norm.logpdf(X[k], loc=mus, scale=sigmas.reshape((sigmas.size, 1)))

        return out


    def _construct_link_matrix(self, K):
        G = np.empty((K, K), dtype=object)

        for i in range(K):
            for j in range(K):
                if i == j:
                    G[i, j] = ''
                else:
                    G[i, j] = f'{i}->{j}'
        
        return G

    """
    Constructs a 2D array containing all the value combinations for a set of parameter labels
    Crucial for quicker updates.
    """
    def _construct_parameter_combinations(self, param_labels, param_values):
        num_values = np.array([p_v.size for p_v in param_values])

        c = len(param_labels) # Number of parameters (columns)
        r = num_values.prod() # Number of tuples (rows)
        S = np.zeros((r, c))
        for i in range(c):
            tile_coef = int(num_values[i+1:].prod())
            rep_coef = int(num_values[:i].prod())
            ou = np.tile(param_values[i], (int(tile_coef), 1)).flatten('F')
            o = ou[np.tile(np.arange(ou.size), (1, rep_coef)).flatten()]
            S[:, i] = o.T

        return S
        
        
    """
    Constructs a dictionary
    keys: the relevant parameter subsets needed for update as stringified lists
    values: the parameter value space as nd.array with row is a combination and columns are parameters in order given by the key 
    """
    def _construct_parameter_subset_dict(self):
        subsets = {}

        # Links
        for k in self._link_params:
            sub = self._no_link_params + [k]
            values = [self._param_names_dict for p in sub]
            subsets[','.join(sub)] = self._construct_parameter_combinations(sub, values)

        # Interventions
        for nl in self._no_link_params:
            sub = [nl] + self._link_params
            values = [self._param_names_dict for p in sub]
            subsets[','.join(sub)] = self._construct_parameter_combinations(sub, values)
            for a in range(self._K):
                sub = [nl] + np.delete(self._link_names_matrix[a, :], a).tolist()
                values = [self._param_names_dict for p in sub]
                subsets[','.join(sub)] = self._construct_parameter_combinations(sub, values)

        return subsets

        