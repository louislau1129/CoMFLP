import os
import pdb
import numpy as np

from itertools import combinations
import copy

import sys
sys.path.insert(0, "../")

from clip_finetune_train import ExtendedWav2Vec2ForCTC

from utils import set_seed, setup_logger, relocate, str2bool

from search_pruning import regulate_clip_layers_list, greedy_search_onestep

import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
from transformers import Wav2Vec2Model
import pdb
import sys


from data import SpeechDataset, SpeechCollate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os

import numpy as np
import time

import torch.nn as nn

from datasets import load_metric
import argparse

from copy_clipped_model import downlayer_copy_weight_series

from DC import distance_matrix, distance_correlation_from_distmat
import copy
import logging

from search_pruning import beam_search_onestep


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--num_layers", type=int, default=12, required=True)
    parser.add_argument("--num_clip_layers", type=int, default=2, required=True)
    
    parser.add_argument("--search_mode", type=str, default="exhaustive", required=True,
                        choices=["exhaustive", "greedy", "beam"])

    parser.add_argument("--select_measure", type=str, default="dc", required=True,
                        choices=["dc", "svcca", "cosine", "covcorr", "delta_mean", 'rand', "combine"],
                        help="measure can be dc mat or cca mat, even others")
      

    parser.add_argument("--topk_num", type=int, default=10, required=False,
                            help="in exhaustive consider how many proposals of layer pruning")
    parser.add_argument("--verify_wer", type=str2bool, default=True)
    parser.add_argument("--verify_dc", type=str2bool, default=True)
    parser.add_argument("--lastlayer_isctc", type=str2bool, default=False)
    parser.add_argument("--test_num_batch", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=10)

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--mode", type=str, default="mean",
                help="DC measure: mean or seq")
    parser.add_argument("--svcca_mode",type=str, default="T",
                help="can be T or U, U means utterance-level")
    parser.add_argument("--thre", type=float, default=0.99, 
                help="the thre percentage of singular values", required=False)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    
    if args.select_measure == "svcca":
        log_name = f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_search_then_verify_SVCCAmix.log"
    elif args.select_measure == "cosine":
        log_name = f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_search_then_verify_COSINEmix.log"
    elif args.select_measure == "covcorr":
        log_name = f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_search_then_verify_COVCORRmix.log"
    elif args.select_measure == "delta_mean":
        log_name = f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_search_then_verify_DELTAMEANmix.log"
    elif args.select_measure == "rand":
        log_name = f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_search_then_verify_RANDmix.log"
    else:
        log_name = f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_search_then_verify_bs{args.bs}_iter{args.iter}.log"
    setup_logger(log_name=log_name, save_dir='log')

    num_layers = args.num_layers
    layers = np.arange(1, num_layers+1)

    # loading the pre-computed dc matrix
    logging.info("Loading the corresponding DC matrix between layers of original model")
    logging.info(f"Note this DC matrix is actually {args.select_measure.upper()} matrix")
    if args.num_layers == 12:
        if args.select_measure == "dc" or args.select_measure == "rand":
            # dc_mat = np.load("../dump/dc_mat_mean_100.npy")    
            dc_mat = np.load(f"../dump/dc_mat_{args.mode}_bs{args.bs}_iter{args.iter}.npy")    
        elif args.select_measure == "svcca":
            dc_mat = np.load(f"../dump/cca_mat_{args.bs * args.iter}{args.svcca_mode}_svthre{args.thre}.npy") 
        elif args.select_measure == "cosine":
            dc_mat = np.load("../dump/dir_corr_mat_100.npy")
        elif args.select_measure == "covcorr":
            dc_mat = np.load("../dump/cov_corr_mat_100.npy")
        elif args.select_measure == "delta_mean":
            dc_mat = np.load("../dump/delta_mean_mat.npy")
            dc_mat = -dc_mat
        elif args.select_measure == "combine":
            dc_mat = np.load("../dump/dc_cca_mat_combine.npy")
        else:
            raise RuntimeError(f"Not suported measure: {args.select_measure}")
    
    elif args.num_layers == 6:
        # dc_mat = np.load("dump/dc_mat_mean_10_numlayer6.npy")    
        dc_mat = np.load(f"../dump_6/dc_mat_{args.mode}_bs{args.bs}_iter{args.iter}.npy")    
    else:
        raise RuntimeError(f'Not supported such num_layers {args.num_layers}')
    
    num_clip_layers = args.num_clip_layers

    logging.info(f"Layer Pruning {num_clip_layers} layers from {num_layers} layers")

    if args.search_mode == "exhaustive":
        clip_layers_comb = list(combinations(layers, num_clip_layers))
        logging.info(f"Exhausitive search {len(clip_layers_comb)} pruning ways")

        dc_measure_list = []
        for clip_layers in clip_layers_comb:
            # logging.info('Before regulated: ', clip_layers)
            regulated_clip_layers = regulate_clip_layers_list(clip_layers)
            # logging.info('After regulated: ', regulated_clip_layers)

            dc = 0
            for item in regulated_clip_layers:
                i = item[0] - 1
                j = item[1]
                dc += dc_mat[i][j]
            dc = dc / len(regulated_clip_layers)
            dc_measure_list.append(dc)

        assert len(dc_measure_list) == len(clip_layers_comb)

        if not args.select_measure == "rand":
            idxes = np.argsort(dc_measure_list)
            topk_idx =  idxes[-args.topk_num:]
            topk_idx = topk_idx[::-1]
        else:
            topk_idx = np.random.choice(np.arange(len(clip_layers_comb)), 
                        args.topk_num, replace=False)
       
        # lastk_idx = idxes[:args.topk_num]
        # lastk_idx = lastk_idx[::-1]
        # topk_idx = np.concatenate((topk_idx, lastk_idx))
        for i, idx in enumerate(topk_idx):
            regulated_clip = regulate_clip_layers_list(clip_layers_comb[idx])
            logging.info(f"{i+1}-th layer prunning (dc_measure: {dc_measure_list[idx]}): {regulated_clip}")
    
    elif args.search_mode == "greedy":
        logging.info(f"Greedy search pruning ways")
        prior_clipped_layers = []
        remained_layers = []
        for i in range(1, num_layers+1):
            remained_layers.append(i)
        
        for i in range(1, num_clip_layers+1):
            prior_clipped_layers, remained_layers = greedy_search_onestep(
                prior_clipped_layers, remained_layers, measure_mat=dc_mat)
            logging.info(f"prune {i} layer,  prior_clipped_layers: {regulate_clip_layers_list(prior_clipped_layers)}")
    
    elif args.search_mode == "beam":
        if args.select_measure != "rand":
            print(f"Beam search pruning ways")
            prior_clipped_layers_list = [[]]
            remained_layers_list = [[]]
            for i in range(1, num_layers+1):
                remained_layers_list[0].append(i)

            for i in range(1, num_clip_layers+1):
                prior_clipped_layers_list, remained_layers_list, top_measure = beam_search_onestep(
                    prior_clipped_layers_list, remained_layers_list, measure_mat=dc_mat, beam_size=args.beam_size
                )

            for x in prior_clipped_layers_list:
                print(f"[BEAM_SIZE:{args.beam_size}] prune {i} layer,  prior_clipped_layers: {regulate_clip_layers_list(x)}")
        else: 
            clip_layers_comb = list(combinations(layers, num_clip_layers))
            topk_idx = np.random.choice(np.arange(len(clip_layers_comb)), 
                        args.topk_num, replace=False)
            prior_clipped_layers_list = [clip_layers_comb[i] for i in topk_idx]
            
            regulated_clipped_layers_list = [[[1,6]], [[2,7]], [[3,8]], [[4,9]], [[5,10]], [[6,11]], [[7,12]]]
            regulated_clipped_layers_list = [[[1,3]], [[2,4]], [[5,7]], [[9,11]], [[10,12]]]
            regulated_clipped_layers_list = [[[1,1]], [[6,6]], [[7,7]], [[12,12]]]
            regulated_clipped_layers_list = [[[1,2]], [[5,6]], [[6,7]], [[11,12]]]
            regulated_clipped_layers_list = [[[1,3]], [[5,7]], [[10,12]]]
            regulated_clipped_layers_list = [[[1,4]], [[4,7]], [[5,8]], [[9,12]]]
            regulated_clipped_layers_list = [[[1,5]], [[4,8]], [[8,12]]]
            regulated_clipped_layers_list = [[[1,6]], [[3,8]], [[4,9]], [[7,12]]]
            top_measure = []
            for clip_layers in prior_clipped_layers_list:
                pass
                # for regulated_clip_layers in regulated_clipped_layers_list:
                regulated_clip_layers = regulate_clip_layers_list(clip_layers)
                # logging.info('After regulated: ', regulated_clip_layers)

                dc = 0
                for item in regulated_clip_layers:
                    i = item[0] - 1
                    j = item[1]
                    dc += dc_mat[i][j]
                dc = dc / len(regulated_clip_layers)
                top_measure.append(dc)
            # top_measure = np.random.randn(args.topk_num)
            top_measure = np.array(top_measure)

    else:
        raise RuntimeError(f"Not supported such search_mode: {args.search_mode}")
    

    
    logging.info("--- Corase layer pruning search finished, Now enter into the verify stage")
    # we have two methods to fine-grained verify before start real fine-tuning
    # (1) test WER on a small set of labeled test data
    # (2) compute the final layer dc between clipped model and orginal model, see its correlation with WER

    logging.info("--- Construct the dataset")
    # wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/wav.scp"
    # text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/text"
    text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/text"

    # wav_scp = "/home/louislau/research/prep_mfa/data/CSRC2021_RAW/SetC1/wav.scp"
    # text_file = "/home/louislau/research/prep_mfa/data/CSRC2021_RAW/SetC1/text"

    dataset = SpeechDataset(wav_scp=wav_scp, text_file=text_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=SpeechCollate())

    pretrained_path_name = "kehanlu/mandarin-wav2vec2-aishell1"
    config = Wav2Vec2Config.from_pretrained(pretrained_path_name)
    if args.verify_dc:
        config.output_hidden_states = True
    # pdb.set_trace()

    logging.info(f"-- construct the original {args.num_layers}-layer model")
    if args.num_layers == 12:
        assert config.num_hidden_layers == args.num_layers
        model =ExtendedWav2Vec2ForCTC(config=config)
        model = model.from_pretrained(pretrained_path_name)
        if args.verify_dc:
            model.config.output_hidden_states = True
    elif args.num_layers == 6:
        config.num_hidden_layers = args.num_layers
        model = ExtendedWav2Vec2ForCTC(config=config)
        pt_path = "../exp/clip12to6_ft_simple/asr_ep20.pt" 
        pt_path = "../exp/clip12to6/clip12to6.pt" 
        model.load_state_dict(torch.load(pt_path))
        logging.info(f"--load {pt_path} for 6 layer model")
    else:
        raise RuntimeError(f"Not supported {args.num_layers} at this time.")
    
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    processor = Wav2Vec2Processor.from_pretrained(pretrained_path_name)
    wer_metric = load_metric("wer")  
    
    if args.search_mode == "exhaustive":
        WERS = []
        LastLayerDCs = []
        DCsfirststage = []
        for i, idx in enumerate(topk_idx):
            DCsfirststage.append(dc_measure_list[idx])

            regulated_clip = regulate_clip_layers_list(clip_layers_comb[idx])
            logging.info(f"{i+1}-th layer prunning (dc_measure: {dc_measure_list[idx]}): {regulated_clip}")

            # clipped model according the suggested layer pruning strategy
            new_config = model.config
            new_config.num_hidden_layers = args.num_layers - args.num_clip_layers
            # new_config.update({'connect_layers': [0, 2]})

            logging.info("--- Construct the clipped model")
            clipped_model = ExtendedWav2Vec2ForCTC(config=new_config)

            # logging.info("--- Start copying weight from deepnet to shallownet")
            deepnet_statedict = model.state_dict()
            clipped_model.load_state_dict(deepnet_statedict, strict=False)
            
            # series_clipped_layers= [[2, 2], [4, 4], [6, 6]]
            # series_clipped_layers=[[2,4], [7,8], [11,11]]
            # series_clipped_layers=[[4, 6]]
            series_clipped_layers = copy.deepcopy(regulated_clip)
            # logging.info(f"series_clipped_layers: {series_clipped_layers}") 
            
            clip_num = 0
            for clip_layer in series_clipped_layers:
                clip_num += clip_layer[1] - clip_layer[0] +1
            assert clip_num == args.num_clip_layers

            shallownet_statedict = clipped_model.state_dict()
            shallownet_statedict =  downlayer_copy_weight_series(
                deepnet_statedict, shallownet_statedict,
                series_clipped_layers=series_clipped_layers
            )
            clipped_model.load_state_dict(shallownet_statedict, strict=True)

            clipped_model.eval()
            clipped_model.to(device)

            ASR_Trans = []
            Clipped_ASR_Trans = []
            Label_Trans = [] 

            sample_rate = 16000
            start_time = time.time()

            dc = 0
            dc_logit = 0
            for i, batch in enumerate(tqdm(dataloader)):
                if i >= args.test_num_batch:
                    break
                audio_input = batch['audio'] 
                inputs = processor(audio_input, sampling_rate=sample_rate, 
                                    padding=True, return_tensors="pt",
                                    return_attention_mask=True,
                                        dtype=torch.float32)
                inputs['input_values'] = inputs['input_values'].float()
                inputs = relocate(inputs, device)
                with torch.no_grad():
                    clipped_model_output = clipped_model(**inputs)
                    model_output = model(**inputs)
                    if args.verify_dc:
                        if not args.lastlayer_isctc:
                            last_layer = model_output.hidden_states[-1]
                            clipped_last_layer = clipped_model_output.hidden_states[-1]
                        else:
                            last_layer = model_output.logits
                            clipped_last_layer = clipped_model_output.logits
                        # [B, T, D] --> [B, D]
                        last_layer = last_layer.mean(dim=1)
                        clipped_last_layer = clipped_last_layer.mean(dim=1)

                        dc += distance_correlation_from_distmat(
                            distance_matrix(last_layer), distance_matrix(clipped_last_layer))
                        
                    if args.verify_wer:
                        clipped_logits = clipped_model_output.logits
                        # logits = model_output.logits
                        # hidden_states = model_output.hidden_states
                        clipped_predicted_ids = torch.argmax(clipped_logits, dim=-1)
                        # predicted_ids = torch.argmax(logits, dim=-1)
                        clipped_transcription = processor.batch_decode(clipped_predicted_ids)
                        # transcription = processor.batch_decode(predicted_ids)
                        label_transcription = batch['text']

                        # seems wer here need to separated by space
                        # transcription = [" ".join(t) for t in transcription]
                        clipped_transcription = [" ".join(t) for t in clipped_transcription]
                        label_transcription = [" ".join(t) for t in label_transcription]

                        # ASR_Trans += transcription
                        Clipped_ASR_Trans += clipped_transcription
                        Label_Trans += label_transcription
            # logging.info(f"Decoding Elapsed {time.time() - start_time}")
            if args.verify_dc:
                dc /= args.test_num_batch
                LastLayerDCs.append(dc.cpu().item())
                logging.info(f"For {regulated_clip}: final layer dc vs. original: {dc}") 

            if args.verify_wer:
                # logging.info(f"-- WER stats on {len(Label_Trans)} utterances")
                clipped_wer = wer_metric.compute(predictions=Clipped_ASR_Trans,
                                                references=Label_Trans) 
                logging.info(f"For {regulated_clip}: clipped wer : {clipped_wer*100}%") 
                WERS.append(clipped_wer)

        logging.info(f"DCsfirststage: {DCsfirststage}")
        if args.verify_wer:        
            logging.info(f"WERs: {WERS}")
        if args.verify_dc:
            logging.info(f"LastLayerDCs: {LastLayerDCs}")  
        
        pcc1 = np.corrcoef(WERS, DCsfirststage)[0][1]
        logging.info(f"correlation between wer and firststage_dc: {pcc1} ")


        pcc2 = np.corrcoef(WERS, LastLayerDCs)[0][1]
        logging.info(f"correlation between wer and lastlayer_dc: {pcc2} ")

        logging.info("Plotting the stage1_dc and stage2_dc vs. WER relationship")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import seaborn as sns

        WERS = np.array(WERS) * 100
        DCsfirststage = np.array(DCsfirststage)
        LastLayerDCs = np.array(LastLayerDCs)
        
        a1, b1 = np.polyfit(WERS, DCsfirststage, deg=1)
        a2, b2 = np.polyfit(WERS, LastLayerDCs, deg=1)
        
        plt.figure()
        plt.scatter(WERS, DCsfirststage, marker='o')
        plt.plot(WERS, a1*WERS+b1)
    
        plt.scatter(WERS, LastLayerDCs, marker="^")
        plt.plot(WERS, a2*WERS+b2)

        if args.select_measure == "dc" or args.select_measure == "rand":
            plt.legend(["Stage1_DC", "Stage2_DC"])
        elif args.select_measure == "svcca":
            plt.legend(["Stage1_SVCCA", "Stage2_DC"])
        elif args.select_measure == "cosine":
            plt.legend(["Stage1_COSINE", "Stage2_DC"])
        elif args.select_measure == "covcorr":
            plt.legend(["Stage1_COVCORR", "Stage2_DC"])
        elif args.select_measure == "delta_mean":
            plt.legend(["Stage1_DELATMEAN", "Stage2_DC"])




        plt.xlabel("Word Error Rate (%)")
        plt.ylabel("DC value")

        if not args.lastlayer_isctc:
            if args.select_measure == "svcca":
                plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_svcca")
                plt.savefig(f"figure/wer_svcca2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.png")
            elif args.select_measure == "cosine":
                plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_cosine")
                plt.savefig(f"figure/wer_cosine2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.png")
            elif args.select_measure == "covcorr":
                plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_covcorr")
                plt.savefig(f"figure/wer_covcorr2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.png")
            elif args.select_measure == "delta_mean":
                plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_deltamean")
                plt.savefig(f"figure/wer_deltamean2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.png")
            elif args.select_measure == "rand":
                plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_rand")
                plt.savefig(f"figure/wer_rand2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.png")
            
            else:
                plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers}")
                plt.savefig(f"figure/wer_2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.png")
        else:
            plt.title(f"clip{args.num_layers}to{args.num_layers - args.num_clip_layers} lastlayer:CTC")
            plt.savefig(f"figure/wer_2stagedc_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}_CTC.png")


    elif args.search_mode == "greedy":
        regulated_clip = regulate_clip_layers_list(prior_clipped_layers)
        # clipped model according the suggested layer pruning strategy
        new_config = model.config
        new_config.num_hidden_layers = args.num_layers - args.num_clip_layers
        # new_config.update({'connect_layers': [0, 2]})
        logging.info("--- Construct the clipped model")
        clipped_model = ExtendedWav2Vec2ForCTC(config=new_config)
        # logging.info("--- Start copying weight from deepnet to shallownet")
        deepnet_statedict = model.state_dict()
        clipped_model.load_state_dict(deepnet_statedict, strict=False)
        
        # series_clipped_layers= [[2, 2], [4, 4], [6, 6]]
        # series_clipped_layers=[[2,4], [7,8], [11,11]]
        # series_clipped_layers=[[4, 6]]
        series_clipped_layers = copy.deepcopy(regulated_clip)
        # logging.info(f"series_clipped_layers: {series_clipped_layers}") 
        
        clip_num = 0
        for clip_layer in series_clipped_layers:
            clip_num += clip_layer[1] - clip_layer[0] +1
        assert clip_num == args.num_clip_layers
        shallownet_statedict = clipped_model.state_dict()
        shallownet_statedict =  downlayer_copy_weight_series(
            deepnet_statedict, shallownet_statedict,
            series_clipped_layers=series_clipped_layers
        )
        clipped_model.load_state_dict(shallownet_statedict, strict=True)
        clipped_model.eval()
        clipped_model.to(device)
        ASR_Trans = []
        Clipped_ASR_Trans = []
        Label_Trans = [] 
        sample_rate = 16000
        start_time = time.time()
        dc = 0
        dc_logit = 0
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= args.test_num_batch:
                break
            audio_input = batch['audio'] 
            inputs = processor(audio_input, sampling_rate=sample_rate, 
                                padding=True, return_tensors="pt",
                                return_attention_mask=True,
                                    dtype=torch.float32)
            inputs['input_values'] = inputs['input_values'].float()
            inputs = relocate(inputs, device)
            with torch.no_grad():
                clipped_model_output = clipped_model(**inputs)
                model_output = model(**inputs)
                if args.verify_dc:
                    if not args.lastlayer_isctc:
                        last_layer = model_output.hidden_states[-1]
                        clipped_last_layer = clipped_model_output.hidden_states[-1]
                    else:
                        last_layer = model_output.logits
                        clipped_last_layer = clipped_model_output.logits
                    # [B, T, D] --> [B, D]
                    last_layer = last_layer.mean(dim=1)
                    clipped_last_layer = clipped_last_layer.mean(dim=1)
                    dc += distance_correlation_from_distmat(
                        distance_matrix(last_layer), distance_matrix(clipped_last_layer))
                    
                if args.verify_wer:
                    clipped_logits = clipped_model_output.logits
                    # logits = model_output.logits
                    # hidden_states = model_output.hidden_states
                    clipped_predicted_ids = torch.argmax(clipped_logits, dim=-1)
                    # predicted_ids = torch.argmax(logits, dim=-1)
                    clipped_transcription = processor.batch_decode(clipped_predicted_ids)
                    # transcription = processor.batch_decode(predicted_ids)
                    label_transcription = batch['text']
                    # seems wer here need to separated by space
                    # transcription = [" ".join(t) for t in transcription]
                    clipped_transcription = [" ".join(t) for t in clipped_transcription]
                    label_transcription = [" ".join(t) for t in label_transcription]
                    # ASR_Trans += transcription
                    Clipped_ASR_Trans += clipped_transcription
                    Label_Trans += label_transcription
        clipped_wer = wer_metric.compute(predictions=Clipped_ASR_Trans,
                                        references=Label_Trans) 
        logging.info(f"For {regulated_clip}: clipped wer : {clipped_wer*100}%")


    elif args.search_mode == "beam":
        WERS = []
        LastLayerDCs = []
        for i in range(len(prior_clipped_layers_list)):

            regulated_clip = regulate_clip_layers_list(prior_clipped_layers_list[i])
            # for regulated_clip in regulated_clipped_layers_list:
            # clipped model according the suggested layer pruning strategy
            new_config = model.config
            new_config.num_hidden_layers = args.num_layers - args.num_clip_layers
            # new_config.update({'connect_layers': [0, 2]})

            logging.info("--- Construct the clipped model")
            clipped_model = ExtendedWav2Vec2ForCTC(config=new_config)

            # logging.info("--- Start copying weight from deepnet to shallownet")
            deepnet_statedict = model.state_dict()
            clipped_model.load_state_dict(deepnet_statedict, strict=False)
            
            # series_clipped_layers= [[2, 2], [4, 4], [6, 6]]
            # series_clipped_layers=[[2,4], [7,8], [11,11]]
            # series_clipped_layers=[[4, 6]]
            series_clipped_layers = copy.deepcopy(regulated_clip)
            # logging.info(f"series_clipped_layers: {series_clipped_layers}") 
            
            clip_num = 0
            for clip_layer in series_clipped_layers:
                clip_num += clip_layer[1] - clip_layer[0] +1
            assert clip_num == args.num_clip_layers

            shallownet_statedict = clipped_model.state_dict()
            shallownet_statedict =  downlayer_copy_weight_series(
                deepnet_statedict, shallownet_statedict,
                series_clipped_layers=series_clipped_layers
            )
            clipped_model.load_state_dict(shallownet_statedict, strict=True)

            clipped_model.eval()
            clipped_model.to(device)

            ASR_Trans = []
            Clipped_ASR_Trans = []
            Label_Trans = [] 

            sample_rate = 16000
            start_time = time.time()

            dc = 0
            dc_logit = 0
            for i, batch in enumerate(tqdm(dataloader)):
                if i >= args.test_num_batch:
                    break
                audio_input = batch['audio'] 
                inputs = processor(audio_input, sampling_rate=sample_rate, 
                                    padding=True, return_tensors="pt",
                                    return_attention_mask=True,
                                        dtype=torch.float32)
                inputs['input_values'] = inputs['input_values'].float()
                inputs = relocate(inputs, device)
                with torch.no_grad():
                    clipped_model_output = clipped_model(**inputs)
                    model_output = model(**inputs)
                    if args.verify_dc:
                        if not args.lastlayer_isctc:
                            last_layer = model_output.hidden_states[-1]
                            clipped_last_layer = clipped_model_output.hidden_states[-1]
                        else:
                            last_layer = model_output.logits
                            clipped_last_layer = clipped_model_output.logits
                        # [B, T, D] --> [B, D]
                        last_layer = last_layer.mean(dim=1)
                        clipped_last_layer = clipped_last_layer.mean(dim=1)

                        dc += distance_correlation_from_distmat(
                            distance_matrix(last_layer), distance_matrix(clipped_last_layer))
                        
                    if args.verify_wer:
                        clipped_logits = clipped_model_output.logits
                        # logits = model_output.logits
                        # hidden_states = model_output.hidden_states
                        clipped_predicted_ids = torch.argmax(clipped_logits, dim=-1)
                        # predicted_ids = torch.argmax(logits, dim=-1)
                        clipped_transcription = processor.batch_decode(clipped_predicted_ids)
                        # transcription = processor.batch_decode(predicted_ids)
                        label_transcription = batch['text']

                        # seems wer here need to separated by space
                        # transcription = [" ".join(t) for t in transcription]
                        clipped_transcription = [" ".join(t) for t in clipped_transcription]
                        label_transcription = [" ".join(t) for t in label_transcription]

                        # ASR_Trans += transcription
                        Clipped_ASR_Trans += clipped_transcription
                        Label_Trans += label_transcription
            # logging.info(f"Decoding Elapsed {time.time() - start_time}")
            if args.verify_dc:
                dc /= args.test_num_batch
                LastLayerDCs.append(dc.cpu().item())
                logging.info(f"For {regulated_clip}: final layer dc vs. original: {dc}") 

            if args.verify_wer:
                # logging.info(f"-- WER stats on {len(Label_Trans)} utterances")
                clipped_wer = wer_metric.compute(predictions=Clipped_ASR_Trans,
                                                references=Label_Trans) 
                logging.info(f"For {regulated_clip}: clipped wer : {clipped_wer*100}%") 
                WERS.append(clipped_wer)
        logging.info(f'WERS: {WERS}')
        WERS = np.array(WERS) * 100
        logging.info(f"mean WERS: {np.mean(WERS)}, std WERS: {np.std(WERS)}")
        logging.info(f"max WERS: {np.max(WERS)}, min WERS: {np.min(WERS)}")

        pcc1 = np.corrcoef(WERS, top_measure)[0][1]
        logging.info(f"correlation between wer and correlation measure: {pcc1} ")
        
        pcc1 = np.corrcoef(WERS, LastLayerDCs)[0][1]
        logging.info(f"correlation between wer and LastLayerDCs: {pcc1} ")


