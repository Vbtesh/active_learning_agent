from math import log
from os import name
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from copy import deepcopy


class Agent():
    def __init__(self, N, sensory_state, internal_state, action_state, range_values=(-100, 100)):
        # Set parameters
        self._N = N
        self._n = 0

        # SHOULD BE CHANGED, AGENT SHOULD BE OU AGENT
        self._range = range_values

        # Set blanket states
        ## Sensory state: must be a Sensory_state object
        self._sensory_state = sensory_state
        
        ## Action states: must be an Action_state object
        self._action_state = action_state

        # Internal states: must be an Internal_state object
        self._internal_state = internal_state

        # Log likelihood
        self._log_likelihood = 0
        self._log_likelihood_history = np.zeros(self._N + 1)


        
    # Core methods
    ## Learn
    def learn(self, external_state, intervention=None):
        # Observe new external state
        self._sensory_state.observe(external_state, self._internal_state)

        # Update internal states
        self._internal_state.update(self._sensory_state, intervention=intervention)

        self._n += 1

    ## Learn fit 
    def fit_learn(self, external_state, intervention=None):
        # Observe new external state
        self._sensory_state.observe(external_state, self._internal_state)

        # Update internal states
        log_prob_judgement = self._internal_state.update(self._sensory_state, intervention=intervention)

        self._log_likelihood += log_prob_judgement
        self._n += 1

    ## Act by sampling an action
    def act(self, external_state):
        # Sample new action
        action = self._action_state.sample(deepcopy(external_state), 
                                           deepcopy(self._sensory_state), 
                                           deepcopy(self._internal_state))

        return action

    # Fit action
    def fit_action(self, external_state):

        log_prob_action = self._action_state.fit(deepcopy(external_state), 
                                                 deepcopy(self._sensory_state), 
                                                 deepcopy(self._internal_state))

        self._log_likelihood_history[self._n] = self._log_likelihood
        self._log_likelihood += log_prob_action
        return log_prob_action

    # Fit judgement
    def fit_judgement(self, judgement):
        log_prob_judgement = self._internal_state.posterior_PMF(judgement, log=True)
        self._log_likelihood_history[self._n] = self._log_likelihood
        self._log_likelihood += log_prob_judgement


    # Resets the agent by rolling back all states
    def reset(self):
        self._sensory_state.rollback()
        self._internal_state.rollback()
        self._action_state.rollback()

    # Properties
    @property
    def a(self):
        return self._action_state.a
    
    @property
    def states(self):
        return self._sensory_state, self._internal_state, self._action_state

    @property
    def sensory_state(self):
        return self._sensory_state

    @property
    def internal_state(self):
        return self._internal_state

    @property
    def action_state(self):
        return self._action_state

    @property
    def log_likelihood(self):
        return self._log_likelihood


    # Reports
    def plot_perceptions(self):
        palette = sns.color_palette() # Set palette
        sns.set_palette(palette)
        
        data_obs = self.sensory_state.obs

        ax = sns.lineplot(data=data_obs[0:self._n+1,:], lw=1.5) # Plot data

        for i in range(self.sensory_state._K):
            # Plot interventions where relevant
            ints = self.action_state._variable_history[0:self._n+1] == i
            if np.sum(ints) == 0:
                continue
            
            x = np.arange(len(ints))
            y1 = self._range[0] * ints
            y2 = self._range[1] * ints
            ax.fill_between(x, y1, y2, color=palette[i], alpha=0.15)

        plt.title('Network Perception')
        plt.ylim(self._range[0], self._range[1])

    # Reports
    def plot_alt_perceptions(self):
        palette = sns.color_palette() # Set palette
        sns.set_palette(palette)
        
        alt_labels = ['0 - alt', '1 - alt', '2 - alt']
        data_obs_alt = self.sensory_state.obs_alt
        ax = sns.lineplot(data=data_obs_alt[0:self._n+1,:], lw=1) # Plot data
        #ax = sns.lineplot(data=data_obs_alt[0:self._n+1,:], lw=0.5) # Plot data

        for i in range(self.sensory_state._K):
            # Plot interventions where relevant
            ints = self.action_state._variable_history[0:self._n+1] == i
            if np.sum(ints) == 0:
                continue
            
            x = np.arange(len(ints))
            y1 = self._range[0] * ints
            y2 = self._range[1] * ints
            ax.fill_between(x, y1, y2, color=palette[i], alpha=0.15)

        plt.title('Network Alternative Perception')
        plt.ylim(self._range[0] / 2, self._range[1]/ 2)
        plt.ylim(self._range[0] / 100, self._range[1]/ 100)

    def plot_entropy_history(self, colour='blue'):
        entropy = self.internal_state.entropy_history
        ax = sns.lineplot(data=entropy[0:self._n+1], lw=1.5, palette=[colour]) # Plot data
        plt.title('Entropy evolution')

    def plot_posterior(self):
        posterior = self.internal_state.posterior_over_models
        ax = sns.lineplot(data=posterior, lw=1.5) # Plot data
        plt.title('Posterior distribution over models')

    def plot_posterior_history(self):
        posterior_history = self.internal_state.posterior_history
        x = np.arange(0, posterior_history.shape[1])
        y = np.arange(0, posterior_history.shape[0])
        Y, X = np.meshgrid(x, y)
        Z = posterior_history

        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50, cmap='viridis')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Models')
        ax.set_zlabel('Likelihood')
        plt.title('Posterior evolution')

        