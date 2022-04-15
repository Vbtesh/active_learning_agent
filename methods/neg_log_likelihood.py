
import numpy as np

def calc_softmax_neg_log_likelihood(temp, dataset, selection):
    softmax_unnorm = np.exp(dataset / temp)
    softmax = softmax_unnorm / softmax_unnorm.sum(axis=0).reshape((1, dataset.shape[1]))

    judgements_likelihood = softmax[selection]

    log_likelihood = np.log(judgements_likelihood).sum()
    return - log_likelihood




        
