import numpy as np
from scipy.spatial.distance import pdist

def normalised_euclidean_distance(ground_truth, posterior):
    euc_dist = 1-pdist(np.stack((ground_truth, posterior)))[0] / np.linalg.norm(abs(np.array(ground_truth)) + 2*np.ones((1, 6)))
    return euc_dist
