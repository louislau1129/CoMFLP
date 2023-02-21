# this script is used for testing the SVCCA

import os
import sys
import numpy as np

from svcca import cca_core
from svcca import pwcca

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
import pdb

def _plot_helper(arr, xlabel, ylabel, fig_name):
    plt.figure()
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    
    plt.savefig(f"svcca/{fig_name}.png")

def find_sv_num(s, thre=0.99):
    '''
    s is a list of singular values
    find the top number of singular values that account for 99% variance
    '''
    total = 0
    Total = np.sum(s)
    for i,x in enumerate(s):
        total += x
        if (total / Total) > thre:
            break
    return i+1


def SVCCA(acts1, acts2, thre=0.99, sv_num=50):
    '''
        acts1: one set of neurons' activation vectors, [num_neurons1, num_samples]
        acts2: another set of neurons' activation vectors, [num_neurons2, num_samples]
        sv_num: this refers to the number of top singular values should be consided. 
    '''
    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)
     
    sv_num1 = find_sv_num(s1, thre)
    sv_num2 = find_sv_num(s2, thre)
    
    svacts1 = np.dot(s1[:sv_num1]*np.eye(sv_num1), V1[:sv_num1])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:sv_num2]*np.eye(sv_num2), V2[:sv_num2])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)
    
    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, 
                                    epsilon=1e-10, verbose=False)
    return svcca_results['cca_coef1']


def PWCCA(acts1, acts2):
    '''
        acts1: one set of neurons' activation vectors, [num_neurons1, num_samples]
        acts2: another set of neurons' activation vectors, [num_neurons2, num_samples]
    '''
    pwcca_mean, w, _ = pwcca.compute_pwcca(acts1, acts2, epsilon=1e-10)    
    return pwcca_mean

def CCA(acts1, acts2):
    '''
        acts1: one set of neurons' activation vectors, [num_neurons1, num_samples]
        acts2: another set of neurons' activation vectors, [num_neurons2, num_samples]
    '''
    results = cca_core.get_cca_similarity(acts1, acts2, 
                        epsilon=1e-10, verbose=False)
    return results["cca_coef"]


if __name__ == "__main__":
    # Toy Example of CCA in action

    topk = 768
    NUM_SAMPLES = 10000
    '''
    Not that NUM_SMAPLES considered should be at least 5-10 times than neuron_dimension
    '''
    # NUM_SAMPLES = 2000
    # assume X_fake has 768 neurons and 
    # we have their activations on 2000 datapoints
    A_fake = np.random.randn(768, NUM_SAMPLES)
    # Y_fake has 1024 neurons with 
    # activations on the same 2000 datapoints
    # Note X and Y do *not* have to have the same number of neurons
    B_fake = np.random.randn(1024, NUM_SAMPLES)
    # B_fake = np.matmul(np.random.randn(768, 768), A_fake)

    # computing CCA simliarty between X_fake, Y_fake
    # We expect similarity should be very low, because the fake activations are not correlated
    
    
    import time
    start_time = time.time()
    results = cca_core.get_cca_similarity(A_fake, B_fake, 
                        epsilon=1e-10, verbose=False)
    print(f"CCA Elapsed: {time.time() - start_time}")
    # correlation coefficients
    _plot_helper(results['cca_coef1'], 
            xlabel="CCA coef idx", ylabel="CCA coef value", fig_name="toy_egs_cca_coef")
    
    cca_coef = results['cca_coef1']
    print(f"mean CCA similarity between two sets of neurons' activations: {np.mean(cca_coef[:topk])}")
    

    start_time = time.time() 
    svcca_coef = SVCCA(A_fake, B_fake) 
    print(f"SVCCA Elapsed: {time.time() - start_time}")
    _plot_helper(svcca_coef, 
        xlabel="SVCCA coef idx", ylabel="SVCCA coef value", fig_name="toy_egs_svcca_coef")
    print(f"mean SVCCA similarity between two sets of neurons' activations: {np.mean(svcca_coef)}")
    
    start_time = time.time()
    pwcca_mean = PWCCA(A_fake, B_fake) 
    print(f"PWCCA Elapsed: {time.time() - start_time}")
    # _plot_helper(svcca_coef, 
    #     xlabel="PWCCA coef idx", ylabel="PWCCA coef value", fig_name="toy_egs_pwcca_coef")
    print(f"mean PWCCA similarity between two sets of neurons' activations: {pwcca_mean}")
    
    # pdb.set_trace()
    print('done')
