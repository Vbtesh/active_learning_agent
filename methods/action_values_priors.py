import numpy as np
from scipy import stats


def action_values_from_mtv_norm(values, max_action_len, max_obs_len, mus, cov, scale=100):
    
    num_values = values.size
    max_action_len = 10
    max_obs_len = 10

    num_actions = num_values * max_action_len * max_obs_len

    action_len = np.arange(max_action_len)
    obs_len = np.arange(max_obs_len)

    sample_space = np.zeros((num_actions, 3))
    prob = np.zeros((num_actions))

    idx = 0
    for i in values:
        for j in action_len:
            for k in obs_len:
                sample_space[idx, :] = [i, j, k]
                idx += 1

    for i in range(num_actions):
        p = stats.multivariate_normal.pdf(sample_space[i, :], mean=mus, cov=cov)
        prob[i] = p

    dist = prob / prob.sum()

    dist_3d = dist.reshape((num_values, max_action_len, max_obs_len))

    action_values = dist_3d * scale

    return dist, dist_3d, action_values, sample_space


