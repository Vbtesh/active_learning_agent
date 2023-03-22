from math import log
from os import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns



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
        self.realised = False
        self.fitting_judgement = False
        if isinstance(internal_state, list):
            self._multi_is = len(internal_state)

            # Log likelihood
            self._log_likelihood = np.zeros(self._multi_is)
            self._log_likelihood_history = np.zeros((self._N + 1, self._multi_is))

            if internal_state[0]._realised:
                self.realised = True
            if internal_state[0]._fitting_judgement:
                self.fitting_judgement = True
        else:
            self._multi_is = 0

            # Log likelihood
            self._log_likelihood = 0
            self._log_likelihood_history = np.zeros(self._N + 1)

            if internal_state._realised:
                self.realised = True
            if internal_state._fitting_judgement:
                self.fitting_judgement = True

  
    # Core methods
    ## Learn
    def learn(self, external_state):
        
        if self._multi_is:
            for is_idx in range(self._multi_is):
                # Observe new external state
                self._sensory_state[is_idx].observe(external_state, self._internal_state[is_idx])
                # Update internal states
                self._internal_state[is_idx].update(self._sensory_state[is_idx], self.action_state)
        else:
            # Observe new external state
            self._sensory_state.observe(external_state, self._internal_state)
            # Update internal states
            self._internal_state.update(self._sensory_state, self.action_state)

        self._n += 1

    ## Learn fit 
    def fit_learn(self, external_state):
        

        if self._multi_is:
            for is_idx in range(self._multi_is):
                # Observe new external state
                self._sensory_state[is_idx].observe(external_state, self._internal_state[is_idx])
                # Update internal states
                log_prob_judgement = self._internal_state[is_idx].update(self._sensory_state[is_idx], self.action_state)
                self._log_likelihood[is_idx] += log_prob_judgement

            # Save LL history
            self._log_likelihood_history[self._n, :] = self._log_likelihood
        else:
            # Observe new external state
            self._sensory_state.observe(external_state, self._internal_state)
            # Update internal states
            log_prob_judgement = self._internal_state.update(self._sensory_state, self.action_state)
            self._log_likelihood += log_prob_judgement

            # Save LL  history
            self._log_likelihood_history[self._n] = self._log_likelihood


        self._n += 1

    ## Act by sampling an action
    def act(self, external_state):
        # Sample new action
        action = self._action_state.sample(external_state, 
                                           self._sensory_state, 
                                           self._internal_state)

        return action

    # Fit action
    def fit_action(self, external_state):

        log_prob_action = self._action_state.fit(external_state, 
                                                 self._sensory_state, 
                                                 self._internal_state)

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

    @property
    def MAP(self):
        if self._multi_is:
            MAP = [IS.MAP for IS in self._internal_state]
            return MAP
        else:
            return self._internal_state.MAP

    @property
    def posterior_entropy(self):
        if self._multi_is:
            posterior_entropies = [IS.posterior_entropy for IS in self._internal_state]
            return posterior_entropies
        else:
            return self._internal_state.posterior_entropy
        

    @property
    def posterior_entropy_unsmoothed(self):
        if self._multi_is:
            posterior_entropies = [IS.posterior_entropy_unsmoothed for IS in self._internal_state]
            return posterior_entropies
        else:
            return self._internal_state.posterior_entropy_unsmoothed

    @property
    def entropy_history(self):
        if self._multi_is:
            entropy_history = [IS.entropy_history for IS in self._internal_state]
            return np.array(entropy_history)
        else:
            return self._internal_state.entropy_history

    @property
    def final_judgement(self):
        if not self.realised:
            return None
        elif self._multi_is:
            return self._internal_state[0]._judgement_final
        else:
            return self._internal_state._judgement_final


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
        plt.ylim(data_obs_alt.min(),data_obs_alt.max())

    def plot_entropy_history(self, colour='blue'):
        if not self._multi_is:
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


    def plot_variational_entropies(self, ax=None, labels=None):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        if labels:
            label_mapping = {str(k):labels[k] for k in range(self.internal_state._K)}
        else:
            label_mapping = {str(k):str(k) for k in range(self.internal_state._K)}
        
        entropies = self.internal_state.variational_posterior_entropy_history

        df = pd.DataFrame(entropies, columns=self.internal_state._param_names_list)

        sns.lineplot(data=df, ax=ax)

        sns.lineplot(x=np.arange(df.shape[0]), y=self.internal_state._epsilon*np.ones(df.shape[0]), color='black', alpha=0.85, ax=ax)

        ax.legend(self._format_labels(df.columns, label_mapping), loc=6, bbox_to_anchor=(1, 0.5), fontsize=15)
        
        ax.set_title(f'Factors entropy histories \n black line represents the certainty threshold ($\epsilon = {self.internal_state._epsilon}$)')

        sns.despine(ax=ax, left=False, bottom=False, trim=False)
        
        return ax

    def plot_variational_schedules(self, ax=None, labels=None):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        if labels:
            label_mapping = {str(k):labels[k] for k in range(self.internal_state._K)}
        else:
            label_mapping = {str(k):str(k) for k in range(self.internal_state._K)}

        hist = self.internal_state._update_schedule_history

        hist_idx = (hist*(np.arange(hist.shape[1])[::-1] + 1)).astype(float)

        hist_idx[np.where(hist_idx == 0)] = np.nan

        df = pd.DataFrame(hist_idx, columns=self.internal_state._param_names_list[::-1])

        sns.lineplot(data=df, ax=ax)
        ax.set_yticks(np.arange(df.columns.size) + 1)
        ax.set_yticklabels(self._format_labels(df.columns, label_mapping))
        ax.legend([], frameon=False)

        ax.set_xlim(0, self._N)

        ax.set_title(f'Update schedules: factors updated at each timepoint')

        sns.despine(ax=ax, left=False, bottom=False, trim=False)

        return ax


    def _format_labels(self, labels, label_mapping):
        new_labels = []
        for l in labels:
            if l[0] not in np.arange(10).astype(str):
                new_labels.append(f'$\{l}$')
            else:
                split_t = l.split('->')
                nt = f'{label_mapping[split_t[0]]} \\rightarrow {label_mapping[split_t[1]]}'
                new_labels.append(f'${nt}$')
        return new_labels
        