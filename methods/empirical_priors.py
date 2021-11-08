import numpy as np
from scipy import stats


def entropy(distribution):
    log_dist = np.log2(distribution, where=distribution!=0)
    if len(distribution.shape) == 1:
        return - np.sum(distribution * log_dist)
    else:
        return - np.sum(distribution * log_dist, axis=1)


def softmax(d, temp=1):
    return np.exp(temp*d) / np.exp(temp*d).sum(axis=1).reshape((d.shape[0], 1))


# Returns the parameters of a discrete empirical prior distribution over links 
## Uses the distance softmax rule and a temperature parameter
def generate_discrete_empirical_priors(prior_judgement, links, temperature):

    p_unnormalised = np.zeros((prior_judgement.size, links.size))
    for i in np.arange(prior_judgement.size):
        v = prior_judgement[i]
        indices = np.arange(links.size)

        # Computes an a array with negative absolute distance from the judgement
        p_unnormalised[i,:] = links.size - np.abs(indices - np.argmin(np.abs(links - v)))

    # Normalises the exponential of the distance times the temperature parameter
    p = softmax(p_unnormalised, temperature)

    # Return the probability distribution and the entropy of the distribution
    return p, entropy(p)


# Returns the paramaters of continuous (Gaussian) empirical prior distribution over links
## The only parameter is the standard deviation or the array of standard deviations (one for each links)
def generate_gaussian_empirical_priors(prior_judgement, sigma):
    means = prior_judgement
    if type(sigma) == np.ndarray:
        sd = sigma
    else:
        sd = sigma * np.ones(means.shape)

    return np.array([means, sd]).T, stats.norm.entropy(means, sigma)