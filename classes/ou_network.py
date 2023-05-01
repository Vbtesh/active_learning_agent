import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


class OU_Network():
    def __init__(self, N, K, dt, theta=0.5, 
                                 sigma=3, 
                                 ground_truth=None,
                                 init_state=None,
                                 range_values=(-100, 100)):
        # Parameters
        if type(ground_truth) == np.ndarray:
            self.set_ground_truth(ground_truth)
        else:
            self._G = None
        
        self._sig = sigma
        self._dt = dt
        self._theta = theta
        self._range = range_values
    
        # State initialisation
        self._N = N         # End of trial, maximum datapoints
        self._n = 0         # Current index in trial
        self._K = K
        self._data_history = []  # List that stores realisations of the network, parameters cannot be changed
        self._inter_history = [] # List that stores interventions on the network, follows the same indexing as data_history

        self._mus = np.zeros((self._N+1, self._K))
        self._self_att = np.zeros((self._N+1, self._K))
        self._mu_att = np.zeros((self._N+1, self._K))

        self._X = np.zeros((N+1, K))
        # If state is non initial, set the first row of the X matrix to be the initial state
        if type(init_state) == np.ndarray:
            self._X[0,:] = init_state
        # Initialise array of empty interventions
        self._I = np.empty(self._N+1)
        self._I[:] = np.nan

        self._realised = False
            
    
    def run(self, iter=1, interventions=None, reset=False):
        if reset:
            self.reset(save=True) # Store last run in history
            
        if self._n > self._N:
            print('Iterations maxed out')
            return
        elif iter > self._N - self._n:
            r_iter = self._N - self._n
        else:
            r_iter = iter

        # Run iterations
        for i in range(r_iter):
            if type(interventions) == np.ndarray:
                intervention = interventions[i]
            elif isinstance(interventions, tuple):
                intervention = interventions
            else:
                intervention = None
            
            self.update(intervention) # Update the network

        # Return the generated values
        return self._X[self._n-r_iter+1:self._n+1, :]


    def update(self, intervention=None):
        # If the data already exists, simply record the action, increase index and return
        if self._realised:
            if isinstance(intervention, tuple) and np.sum(np.isnan(np.array(intervention))) == 0:
                inter_var = int(intervention[0])
                self._I[self._n+1] = inter_var
            self._n +=1
            return

        # Compute attractor
        self_attractor = -1 * self._X[self._n,:] * (np.abs(self._X[self._n,:]) / np.max(self._range))
        causal_attractor = self._X[self._n,:] @ self._G
        att = self_attractor + causal_attractor

        # Store mus
        self._mus[self._n, :] = self._X[self._n,:] + self._theta * self._dt * (att - self._X[self._n,:])
        self._self_att[self._n, :] = self_attractor
        self._mu_att[self._n, :] = causal_attractor

        # Update using a direct sample from a normal distribution
        self._X[self._n+1, :] = np.random.normal(loc=self._X[self._n,:] + self._theta * self._dt * (att - self._X[self._n,:]), scale=self._sig*np.sqrt(self._dt)) 

        # If intervention, set value irrespective of causal matrix
        if isinstance(intervention, tuple) and np.sum(np.isnan(np.array(intervention))) == 0:
            inter_var = int(intervention[0])
            inter_val = intervention[1]
            self._X[self._n+1, inter_var] = inter_val
            self._I[self._n+1] = inter_var

        # Bound values
        self._X[self._n+1, :][self._X[self._n+1, :] < self._range[0]] = self._range[0]
        self._X[self._n+1, :][self._X[self._n+1, :] > self._range[1]] = self._range[1]

        # Record Difference
        # Increment index      
        self._n += 1


    # Load data
    def set_ground_truth(self, ground_truth):
        if len(ground_truth.shape) > 1:
            self._G = ground_truth
        else:
            self._G = self._causality_matrix(ground_truth, fill_diag=1)

    def load_trial_data(self, variables_values):
        self._X = variables_values
        self._N = variables_values.shape[0] - 1

        self._realised = True


    def reset(self, back=np.inf, init_state=None, save=False):
        if save:
            self._data_history.append(self._X[0:self._n+1,:]) # Save data in history
            self._inter_history.append(self._X[0:self._n+1])  # Save interventions in history

        if back > self._N or back > self._n:
            self._n = 0
        else:
            self._n -= int(back)

        if not self._realised:  
            self._X[self._n+1:,:] = 0 # Reset data except for the initial state
            # Reset interventions
            self._I[self._n+1:] = np.nan

            if type(init_state) == np.ndarray:
                self._X[0, :] = init_state


    def plot_network(self, ax=None, labels=None, history=None):
        palette = sns.color_palette() # Set palette
        sns.set_palette(palette)

        if not labels:
            labels = np.arange(self._K)

        for i in range(self._K):
            if ax:
                sns.lineplot(data=self._X[0:self._n+1, i], label=labels[i], ax=ax) # Plot data
            else:
                ax = sns.lineplot(data=self._X[0:self._n+1, i], label=labels[i]) # Plot data

            # Plot interventions where relevant
            ints = self._I[0:self._n+1] == i
            if np.sum(ints) == 0:
                continue
            
            x = np.arange(len(ints))
            y1 = self._range[0] * ints 
            y2 = self._range[1] * ints
            ax.fill_between(x, y1, y2, color=palette[i], alpha=0.15)

        ax.set_title('Network realisation')
        ax.set_ylim(self._range[0], self._range[1])

        # Plot history
        if history:
            pass

        return ax


    # Properties
    @property
    def N(self):
        return self._N

    @property
    def K(self):
        return self._K
    
    @property
    def causal_vector(self):
        return self._causality_vector(self._G)

    @property
    def causal_matrix(self):
        return self._G

    @causal_matrix.setter
    def causal_matrix(self, model):
        self._G = model

    @property
    def sigma(self):
        return self._sig
    
    @property
    def theta(self):
        return self._theta   

    @property
    def data(self):
        return self._X[0:self._n+1,:] 

    @property
    def x(self):
        return self._X[self._n,:]

    @property
    def x_prev(self):
        return self._X[self._n-1,:]

    @property
    def history(self):
        return (self._data_history, self._inter_history)


    # Internal methods
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
