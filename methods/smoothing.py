import numpy as np

# Methods that smoothes a discrete distribution to avoid it converging toward a point mass
## The smoothing is based on two quantities:
### 1. the distance between the argmax and its neighbours 
### 2. a certainty coefficient which is the exp of the negative entropy times a temperature parameter
## It does nothing to a high entropy distribution and a small smoothing (temperature) for low entropy distributions
def smooth(dist, temp=1e-5):
    if len(dist.shape) == 1:
        dist = dist.reshape((1, dist.size))

    indices = np.tile(np.arange(dist.shape[1]), (dist.shape[0], 1))
    max_dist = np.argmax(dist, axis=1).reshape((dist.shape[0], 1))

    smoother = ( dist.shape[1] - np.abs(indices - max_dist) ) / dist.shape[1]

    certainty_coef = np.exp(- entropy(dist)) * temp

    smoothed_values = dist + certainty_coef * smoother

    return smoothed_values / smoothed_values.sum(axis=1).reshape((dist.shape[0], 1))

# Generic smoothing decorator for any discrete update func
def smooth_decorator(update_func, temp):

    def smoothed_update_func(sensory_state, intervention, *update_func_args):
        func = update_func
        posterior_params = func(sensory_state, intervention, *update_func_args)
        smoothed_posterior = smooth(posterior_params, temp)
        print('Entropy:', entropy(posterior_params), entropy(smoothed_posterior))
        return np.log(smoothed_posterior)

    return smoothed_update_func


# Aids methods
def entropy(distribution):
    # Needs to be specifically defined per sub class
    log_dist = np.log2(distribution, where=distribution!=0)
    log_dist[log_dist == -np.inf] = 0

    return - np.sum(distribution * log_dist)
    

def softmax(d, temp=1):
    return np.exp(temp*d) / np.exp(temp*d).sum(axis=1).reshape((d.shape[0], 1))