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


def greedy_search_onestep(prior_clipped_layers,
                        remained_layers, measure_mat):
    '''
    If at the starting point:
        prior_clipped_layers: []
        remained_layers: [1,2,3,4,5,6, 7, 8,9,10,11,12]
    '''
    measure_list = []
    for sel_layer in remained_layers:
        temp = copy.deepcopy(prior_clipped_layers)
        temp.append(sel_layer)
        temp = sorted(temp)
        regulated_temp = regulate_clip_layers_list(temp)
        measure = 0
        for item in regulated_temp:
            i = item[0] - 1
            j = item[1]
            measure += measure_mat[i][j]
        measure = measure / len(regulated_temp)
        measure_list.append(measure)

    top_idx = np.argsort(measure_list)[-1]
    sel_layer = remained_layers[top_idx]
    prior_clipped_layers.append(sel_layer)
    prior_clipped_layers = sorted(prior_clipped_layers)  

    remained_layers.pop(top_idx)
    # print("prior_clipped_layers: ", prior_clipped_layers)
    return prior_clipped_layers, remained_layers  


def beam_search_onestep(prior_clipped_layers_list,
                        remained_layers_list, measure_mat, beam_size=10):
    '''
    If at the starting point:
        prior_clipped_layers_list: [[]]
        remained_layers_list: [[1,2,3,4,5,6, 7, 8,9,10,11,12]]
    '''
    total_layers = len(prior_clipped_layers_list[0]) + len(remained_layers_list[0])
    measure_list = []
    new_clipped_layers_list = []
    for ii, remained_layers in enumerate(remained_layers_list):
        for sel_layer in remained_layers:
            temp = copy.deepcopy(prior_clipped_layers_list[ii])
            temp.append(sel_layer)
            temp = sorted(temp)
            if temp not in new_clipped_layers_list:
                new_clipped_layers_list.append(temp)
            else:
                continue
            regulated_temp = regulate_clip_layers_list(temp)
            measure = 0
            for item in regulated_temp:
                i = item[0] - 1
                j = item[1]
                measure += measure_mat[i][j]
            measure = measure / len(regulated_temp)
            measure_list.append(measure)
    
    if beam_size > 0:
        top_idx = np.argsort(measure_list)[-beam_size:]
    else:
        top_idx = np.argsort(measure_list)[:-beam_size]
    top_idx = top_idx[::-1]

    top_measure = np.array(measure_list)[top_idx]
    return_clipped_layers_list = []
    return_remained_layers_list = []
    for i in top_idx:
        temp = new_clipped_layers_list[i]
        return_clipped_layers_list.append(temp)
        temp_remained = []
        for j in range(1, total_layers+1):
            if j not in temp:
                temp_remained.append(j)
        return_remained_layers_list.append(temp_remained)

    # print("prior_clipped_layers: ", prior_clipped_layers)
    return return_clipped_layers_list, return_remained_layers_list, top_measure




import argparse
def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_layers", type=int, default=12, required=True)
    parser.add_argument("--num_clip_layers", type=int, default=2, required=True)
    
    parser.add_argument("--search_mode", type=str, default="exhaustive",
                        choices=["exhaustive", "greedy", "beam"], required=True)

    parser.add_argument("--select_measure", type=str, default="dc",
                        choices=["dc", "svcca", "cosine", "covcorr", "delta_mean"],
                        help="measure can be dc mat or cca mat, even others")
    parser.add_argument("--beam_size", type=int, default=10)

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--iter", type=int, default=10)
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
            dc_mat = np.load("../dump/cca_mat_320T.npy")
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
    
    num_clip_layers = args.num_clip_layers

    print(f"Layer Pruning {num_clip_layers} layers from {num_layers} layers")

    if args.search_mode == "exhaustive":
        clip_layers_comb = list(combinations(layers, num_clip_layers))
        print(f"Exhausitive search {len(clip_layers_comb)} pruning ways")

        dc_measure_list = []
        for clip_layers in clip_layers_comb:
            # print('Before regulated: ', clip_layers)
            regulated_clip_layers = regulate_clip_layers_list(clip_layers)
            # print('After regulated: ', regulated_clip_layers)

            dc = 0
            for item in regulated_clip_layers:
                i = item[0] - 1
                j = item[1]
                dc += dc_mat[i][j]
            dc = dc / len(regulated_clip_layers)
            dc_measure_list.append(dc)

        assert len(dc_measure_list) == len(clip_layers_comb)
        topk_idx = np.argsort(dc_measure_list)[-10:]
        topk_idx = topk_idx[::-1]
        for i, idx in enumerate(topk_idx):
            regulated_clip = regulate_clip_layers_list(clip_layers_comb[idx])
            print(f"{i+1}-th layer prunning (dc_measure: {dc_measure_list[idx]}): {regulated_clip}")
    
    elif args.search_mode == "greedy":
        print(f"Greedy search pruning ways")
        prior_clipped_layers = []
        remained_layers = []
        for i in range(1, num_layers+1):
            remained_layers.append(i)
        
        for i in range(1, num_clip_layers+1):
            prior_clipped_layers, remained_layers = greedy_search_onestep(
                prior_clipped_layers, remained_layers, measure_mat=dc_mat)
            print(f"prune {i} layer,  prior_clipped_layers: {regulate_clip_layers_list(prior_clipped_layers)}")
    
    elif args.search_mode == "beam":
        print(f"Beam search pruning ways")
        prior_clipped_layers_list = [[]]
        remained_layers_list = [[]]
        for i in range(1, num_layers+1):
            remained_layers_list[0].append(i)
        
        for i in range(1, num_clip_layers+1):
            prior_clipped_layers_list, remained_layers_list = beam_search_onestep(
                prior_clipped_layers_list, remained_layers_list, measure_mat=dc_mat, beam_size=args.beam_size
            )
        
        for x in prior_clipped_layers_list:
            print(f"[BEAM_SIZE:{args.beam_size}] prune {i} layer,  prior_clipped_layers: {regulate_clip_layers_list(x)}")
    else:
        raise RuntimeError(f"Not supported such search_mode: {args.search_mode}")
    
    # pdb.set_trace()
    print('done')
    

    


