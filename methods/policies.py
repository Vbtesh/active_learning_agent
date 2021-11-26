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


def two_d_softmax_policy_init(temperature):

    def two_d_softmax_policy(action_values):
        p = np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))
        return np.random.choice(np.arange(p.size), p=p)

    def pmf_d_step_softmax_policy(action_taken, action_len, action_values):
        action_v = action_taken[1]
        p = np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))
        return p[action_v, action_len-1]

    def params_d_step_softmax_policy(action_values):
        return np.exp(temperature * action_values) / np.sum(np.exp(temperature * action_values))

    return two_d_softmax_policy, pmf_d_step_softmax_policy, params_d_step_softmax_policy
