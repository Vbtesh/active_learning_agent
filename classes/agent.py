import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from copy import deepcopy


class Agent():
    def __init__(self, N, sensory_state, internal_state, action_state):
        # Set parameters
        self._N = N
        self._n = 0

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

        