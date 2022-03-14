import numpy as np


def random_policy(actions):
    return np.random.choice(np.arange(actions.size))


def softmax_policy_init(temperature):

    def softmax_policy(action_values):
        p = np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))
        return np.random.choice(np.arange(p.size), p=p)

    def pmf_softmax_policy(action_taken, action_values):
        p = np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))
        return p[action_taken]

    def params_softmax_policy(action_values):
        return np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))

    return softmax_policy, pmf_softmax_policy, params_softmax_policy


def epsilon_greedy_init(epsilon):

    def e_greedy_policy(action_values):
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(action_values.size))
        else:
            return np.argmax(action_values)

    def pmf_e_greedy_policy(action_taken, action_values):
        if action_taken == np.argmax(action_values):
            return epsilon + (1 - epsilon)/action_values.size
        else:
            return (1 - epsilon)/action_values.size


    def params_e_greedy_policy(action_values):
        params = np.zeros(action_values.shape)
        params += epsilon/params.size
        params[np.argmax(action_values)] += 1 - epsilon
        return params
        
    return e_greedy_policy, pmf_e_greedy_policy, params_e_greedy_policy


def three_d_softmax_policy_init(temperature):

    def three_d_softmax_policy(action_values):
        dims = action_values.shape
        action_idx = np.arange(action_values.size).reshape(dims)

        p = np.exp(temperature * action_values.flatten()) / np.sum(np.exp(temperature * action_values.flatten()))
        choice = np.random.choice(np.arange(p.size), p=p)

        return np.where(action_idx == choice)


    def pmf_three_d_softmax_policy(action_taken, action_values):
        dims = action_values.shape
        p = np.exp(temperature * action_values.flatten()) / np.sum(np.exp(temperature * action_values.flatten()))
        p_3d = p.reshape(dims)

        return p_3d[tuple(action_taken)]

    def params_three_d_softmax_policy(action_values):
        dims = action_values.shape
        p = np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))
        
        return p.reshape(dims)

    return three_d_softmax_policy, pmf_three_d_softmax_policy, params_three_d_softmax_policy


# Any discrete probability distribution without action values over any dimensions
def discrete_policy_init():

    def discrete_policy(distribution):
        dims = distribution.shape
        actions = np.arange(distribution.size).reshape(dims)

        choice = np.random.choice(actions.flatten(), p=distribution.flatten())

        return np.where(actions == choice)


    ## WORK HERE
    def pmf_discrete_policy(action_taken, distribution):
        return distribution[tuple(action_taken)]


    def params_discrete_policy(distribution):
        return distribution

    return discrete_policy, pmf_discrete_policy, params_discrete_policy


