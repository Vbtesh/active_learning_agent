from copy import deepcopy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import external


class Experiment():
    def __init__(self, agent, external_state, num_exp=1):
        self.agent = agent
        self.external_state = external_state

        self._N = self.agent._N
        self._n = 0
        self._iter = num_exp
        self._i = 0

        self._entropy_history = np.zeros((self._iter, self._N))


    def fit(self, console=True):
        if not self.external_state._realised:
            print('Cannot fit, no loaded data, use Experiment.run instead. Exiting...')
            return

        if console:
            print('True model:', self.external_state.causal_vector) 
        self.agent.reset()
        self.external_state.reset()
        self._n = 0

        for n in range(self._N):
            # Collect action and action to fit
            a = self.agent.a

            # Fit action to fit
            self.agent.fit_action(self.external_state)

            # Update external state using action
            self.external_state.run(interventions=a)

            # Learn from the new state
            self.agent.fit_learn(self.external_state)

            if n % 10 == 0 and console:
                print('Iter:', n, 'Current MAP:', self.agent.internal_state.MAP, 'Current LL:', self.agent.log_likelihood, 'Entropy:', self.agent.internal_state.posterior_entropy)

            self._n += 1

        if console:
            print('Iter:', n, 'Current MAP:', self.agent.internal_state.MAP, 'Current LL:', self.agent.log_likelihood, 'Entropy:', self.agent.internal_state.posterior_entropy)
            print('True model:', self.external_state.causal_vector, 'Posterior_judgement:', self.agent.internal_state._judgement_final)
            #print('Final posterior \n', self.agent.internal_state.posterior)
            print('Final log likelihood:', self.agent.log_likelihood)
            print('Final distance:', np.sum((self.agent.internal_state._judgement_final - self.agent.internal_state.MAP)**2))


    def run(self, console=True):     
        self._i = 0

        for i in range(self._iter):
            if console:
                print('True model:', self.external_state.causal_vector) 
                print(f'\n Realisation {i+1} out of {self._iter}: \n')

            self.agent.reset()
            self.external_state.reset()
            self._n = 0

            for n in range(self._N):
                a = self.agent.act(self.external_state)
                x = self.external_state.run(interventions=a)
                self.agent.learn(self.external_state)
                #self.agent.learn(self.external_state, intervention=a)

                #print('Model n:', agent.model._n)
                #print('external_state n:', external_state._n)

                if n % 10 == 0 and console:
                    print('Iter:', n)
                    print('Current MAP:', self.agent.internal_state.MAP, 'Entropy:', self.agent.internal_state.posterior_entropy)
                    #print('Current posterior:')
                    #print(np.around(agent.model.posterior_links, 2))


                self._n += 1

            if console:
                print('Iter:', n)
                print('True model:', self.external_state.causal_vector, 'Final MAP:', self.agent.internal_state.MAP)

            self._entropy_history[i, :] = self.agent.internal_state.entropy_history
            self._i += 1

            
    def entropy_report(self):
        if not self._i <= self._iter:
            print('Call the run method to generate data.')
            return

        if self._iter == 1:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            self.external_state.plot_network()
            plt.subplot(2, 2, 2)
            self.agent.plot_entropy_history()
            plt.subplot(2, 2, 3)
            self.agent.plot_perceptions()
            if self.agent.sensory_state._obs_alt_record:
                plt.subplot(2, 2, 4)
                self.agent.plot_alt_perceptions()

            print('True model:', self.external_state.causal_vector)
            print('Final MAP:', self.agent.internal_state.MAP)
            if self.external_state._realised:
                print('Posterior judgement:', self.agent.internal_state._judgement_final)

        else:
            palette = sns.color_palette("husl", 8)
            for i in range(self._iter):
                sns.lineplot(self._entropy_hist[i,:], palette=palette)


    def change_report(self):
        if not self._i <= self._iter:
            print('Call the run method to generate data.')
            return

        if self._iter == 1:
            # Plot alternate perceptions
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            self.agent.plot_alt_perceptions()







