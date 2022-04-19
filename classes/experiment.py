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
        
        if not agent._multi_is:
            self._entropy_history = np.zeros((self._iter, self.agent._N))
        else:
            self._entropy_history = np.zeros((self._iter, agent._multi_is, self.agent._N))


    def fit(self, console=False, reset=False):
        if not self.external_state._realised and self.agent.realised:
            print('Cannot fit, no loaded data, use Experiment.run instead. Exiting...')
            return

        if console:
            print('True model:', self.external_state.causal_vector) 
        if reset:
            self.agent.reset()
            self.external_state.reset()
        self._n = 0

        for n in range(self._N):
            # Collect action and action to fit
            a = self.agent.a

            # Update external state using action
            self.external_state.run(interventions=a)

            # Learn from the new state
            self.agent.fit_learn(self.external_state)

            # Fit action to agent's action state
            ## Done after as this will define the next action sampled
            self.agent.fit_action(self.external_state)

            if n % 10 == 0 and console:
                print('Iter:', n, 'Current MAP:', self.agent.MAP, 'Current LL:', self.agent.log_likelihood, 'Entropy:', self.agent.posterior_entropy)

            self._n += 1

        if console:
            print('Iter:', n, 'Current MAP:', self.agent.MAP, 'Current LL:', self.agent.log_likelihood, 'Entropy:', self.agent.posterior_entropy)
            if self.agent.fitting_judgement:
                print('True model:', self.external_state.causal_vector, 'Posterior_judgement:', self.agent.final_judgement)
                print('Final log likelihood:', self.agent.log_likelihood)
            print('Final judgement:', self.agent.final_judgement)
            final_distance = np.concatenate(self.agent.MAP).reshape((self.agent._multi_is, 6))
            print('Final distance:', np.sum((self.agent.final_judgement - final_distance)**2, axis=1)**(-1/2))


    def run(self, console=False, reset=False):     
        self._i = 0

        for i in range(self._iter):
            if console:
                print('True model:', self.external_state.causal_vector) 
                print(f'\n Realisation {i+1} out of {self._iter}: \n')

            if reset:
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
                    print('Current MAP:', self.agent.MAP, 'Entropy:', self.agent.posterior_entropy)
                    #print('Current posterior:')
                    #print(np.around(agent.model.posterior_links, 2))


                self._n += 1

            if console:
                print('Iter:', n)
                print('True model:', self.external_state.causal_vector, 'Final MAP:', self.agent.MAP)

            if self.agent._multi_is:
                self._entropy_history[i, :, :] = self.agent.entropy_history
            else:
                self._entropy_history[i, :] = self.agent.entropy_history
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
            self.agent.plot_perceptions()
            if not self.agent._multi_is:
                plt.subplot(2, 2, 3)
                self.agent.plot_entropy_history()
            if self.agent.sensory_state._obs_alt_record:
                plt.subplot(2, 2, 4)
                self.agent.plot_alt_perceptions()

            print('True model:', self.external_state.causal_vector)
            print('Final MAP:', self.agent.MAP)
            if self.agent.realised:
                print('Posterior judgement:', self.agent.final_judgement)

        else:
            palette = sns.color_palette("husl", 8)
            for i in range(self._iter):
                sns.lineplot(self._entropy_history[i,:], palette=palette)


    def change_report(self):
        if not self._i <= self._iter:
            print('Call the run method to generate data.')
            return

        if self._iter == 1:
            # Plot alternate perceptions
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            self.agent.plot_alt_perceptions()







