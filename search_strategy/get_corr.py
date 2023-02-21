import os
import pdb
import numpy as np

from itertools import combinations
import copy

import sys
sys.path.insert(0, "../")

from utils import set_seed, setup_logger

def regulate_clip_layers_list(clip_layers):
    """
    clip_layers: [1,2,3,6,7,9] --> [[1,3], [6,7], [9,9]]
    """
    num_clip_layers = len(clip_layers)
    
    regulated_clip_layers = []
    start_layer = clip_layers[0]
    for i in range(0, num_clip_layers-1):
        if clip_layers[i+1] == clip_layers[i]+1:
            continue
        else:
            regulated_clip_layers.append([start_layer, clip_layers[i]])
            start_layer = clip_layers[i+1]
    
    regulated_clip_layers.append([start_layer, clip_layers[-1]])
    return regulated_clip_layers



import argparse
def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_layers", type=int, default=12, required=False)
    

    parser.add_argument("--select_measure", type=str, default="dc",
                        choices=["dc", "svcca", "cosine", "covcorr", "delta_mean"],
                        help="measure can be dc mat or cca mat, even others")
    parser.add_argument("--beam_size", type=int, default=10)

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--svcca_mode",type=str, default="T",
                help="can be T or U, U means utterance-level")
    parser.add_argument("--thre", type=float, default=0.99, 
                help="the thre percentage of singular values", required=False)
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    num_layers = args.num_layers
    layers = np.arange(1, num_layers+1)

    # loading the pre-computed dc matrix
    if args.num_layers == 12:
        if args.select_measure == "dc":
            # dc_mat = np.load("../dump/dc_mat_mean_100.npy")    
            dc_mat = np.load(f"../dump/dc_mat_mean_bs{args.bs}_iter{args.iter}.npy")    
        elif args.select_measure == "svcca":
            dc_mat = np.load(f"../dump/cca_mat_{args.bs * args.iter}{args.svcca_mode}_svthre{args.thre}.npy") 
        elif args.select_measure == "cosine":
            dc_mat = np.load("../dump/dir_corr_mat_100.npy")
        elif args.select_measure == "covcorr":
            dc_mat = np.load("../dump/cov_corr_mat_100.npy")
        elif args.select_measure == "delta_mean":
            dc_mat = np.load("../dump/delta_mean_mat.npy")
            dc_mat = -dc_mat
        else:
            raise RuntimeError(f"Not suported measure: {args.select_measure}")
    elif args.num_layers == 6:
        dc_mat = np.load("dump/dc_mat_mean_10_numlayer6.npy")  

    elif args.num_layers > 20:
        dc_mat = np.random.randn(args.num_layers+1, args.num_layers+1)  
    else:
        raise RuntimeError(f'Not supported such num_layers {args.num_layers}')
    

    # dc bs32 best wer: lp_wer: 13.14
    series_clipped_layers = [[1,1], [3,3], [5,5], [7,7], [9,9], [11,11]]
    # dc bs32 dc best: lp_wer: 15.36
    # series_clipped_layers = [[1, 1], [3, 4], [7, 7], [9, 9], [11, 11]]
    # wer greedy: lp_wer: 10.39
    # series_clipped_layers = [[1, 1], [4, 5], [7, 7], [10, 10], [12, 12]]
    # dc bs4 dc best wer/best: lp_wer: 12.50%
    # series_clipped_layers = [[2, 3], [5, 5], [7, 7], [9, 9], [11, 11]]
    
    # dc bs256 best wer: 14.56
    series_clipped_layers = [[2, 4], [7, 7], [9, 9], [11, 11]]
    # dc bs32 dc best2 : 44.43
    series_clipped_layers = [[1,2], [4,4], [7,7], [9,9], [11,11]]

    # svcca 11.08%
    series_clipped_layers = [[1, 1], [3, 3], [5, 5], [7, 7], [10, 10], [12, 12]]
    # series_clipped_layers = [[1,6]]
    # series_clipped_layers = [[4,9]]
    # series_clipped_layers = [[7, 12]]
    # series_clipped_layers = [[6, 11]]
    dc = 0
    for item in series_clipped_layers:
        i = item[0] - 1
        j = item[1]
        dc += dc_mat[i][j]
    dc = dc / len(series_clipped_layers)
    print(series_clipped_layers)
    print(f"{args.select_measure} bs:{args.bs} iter: {args.iter}, val:{dc}")
       
    


