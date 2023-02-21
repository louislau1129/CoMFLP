import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import random
import pdb


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def Distance_Correlation(latent, control):
    '''
    latent and control are two
    '''
    # latent = F.normalize(latent)
    # control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)


    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    # pdb.set_trace()
    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r


def distance_matrix(X, normlaize=True):
    '''
    Args: 
        X: [B, D]
    Reutrn:
        Distance_matrix: [B, B], default is the normalized one.
    '''
    X = F.normalize(X)

    matrix_a = torch.sqrt(torch.sum(torch.square(X.unsqueeze(0) - X.unsqueeze(1)), dim = -1) + 1e-12)
    if normlaize:
        matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    else:
        matrix_A = matrix_a
    
    return matrix_A

def distance_correlation_from_distmat(matrix_A, matrix_B):
    '''
    Args:
        matrix_A, matrix_B: [B, B]
    Return:
        distance correlation
    '''
    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r





if __name__ == "__main__":
    SEED = 1234
    print(f"--- set seed as {SEED}")
    set_seed(SEED)

    layer1 = torch.randn(32, 100, 512)
    layer2 = torch.randn(32, 100, 512)

    B, T, D = layer1.shape
    layer1 = layer1.reshape(B, -1)
    layer2 = layer2.reshape(B, -1)

    # layer1 = layer1.reshape(B*T, -1)
    # layer2 = layer2.reshape(B*T, -1)

    dc = Distance_Correlation(layer1, layer2)
    print(f"distance corrrelation between layer1 and layer2: {dc}")
    pdb.set_trace()

    print('done')




