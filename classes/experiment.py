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

        self._entropy_history = np.zeros((self._iter, self._N+1))


    def fit(self, final_judgement):
        if not self.external_state._realised:
            print('Cannot fit data that does not exists, use Experiment.run instead. Exiting...')
            return

        self.agent.reset()
        self.external_state.reset()
        self._n = 0

        for n in range(self._N):
            # Collect action and action to fit
            a = self.external_state.a
            a_fit = self.external_state.a_fit

            #print(a, a_fit)

            # Fit action to fit
            self.agent.fit_action(self.external_state, a, a_fit)

            # Update external state using action
            x = self.external_state.run(interventions=a)

            # Learn from the new state
            self.agent.learn(self.external_state, intervention=a)

            if n % 10 == 0:
                print('Iter:', n, 'Current MAP:', self.agent.internal_state.map, 'Current LL:', self.agent.log_likelihood)

            self._n += 1

        #self.agent.fit_judgement(final_judgement)
        self.agent.fit_judgement(final_judgement)
        print('True model:', self.external_state.causal_vector, 'Posterior:', final_judgement)
        print('Final posterior', self.agent.internal_state.posterior_over_links)
        print('Final log likelihood:', self.agent.log_likelihood)


    def run(self):     
        self._i = 0

        for i in range(self._iter):
            print('True model:', self.external_state.causal_vector) 
            print(f'\n Realisation {i+1} out of {self._iter}: \n')

            self.agent.reset()
            self.external_state.reset()
            self._n = 0

            for n in range(self._N):
                a = self.agent.act(self.external_state)
                x = self.external_state.run(interventions=a)
                self.agent.learn(self.external_state, intervention=a)

                #print('Model n:', agent.model._n)
                #print('external_state n:', external_state._n)

                if n % 10 == 0:
                    print('Iter:', n)
                    print('Current MAP:', self.agent.internal_state.map)
                    #print('Current posterior:')
                    #print(np.around(agent.model.posterior_links, 2))


                self._n += 1

            print('Iter:', n)
            print('Final MAP:', self.agent.internal_state.map)

            self._entropy_history[i, :] = self.agent.internal_state.entropy_history
            self._i += 1

            
    def entropy_report(self):
        if not self._i <= self._iter:
            print('Call the run method to generate data.')
            return

        palette = sns.color_palette("husl", 8)

        if self._iter == 1:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            self.external_state.plot_network()
            plt.subplot(2, 2, 2)
            self.agent.plot_entropy_history()
            plt.subplot(2, 2, 3)
            self.agent.plot_posterior()

            print('True model:', self.external_state.causal_vector)
            print('Final MAP:', self.agent.internal_state.map)
        else:
            for i in range(self._iter):
                sns.lineplot(self._entropy_hist[i,:], palette=palette)




