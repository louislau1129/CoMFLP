import os
import pdb
import numpy as np

from itertools import combinations
import copy

import sys
sys.path.insert(0, "../")

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

from clip_finetune_train import ExtendedWav2Vec2ForCTC
from copy_clipped_model import downlayer_copy_weight_series

from DC import distance_matrix, distance_correlation_from_distmat
import copy
import logging 


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--num_layers", type=int, default=12, required=True)
    parser.add_argument("--num_clip_layers", type=int, default=2, required=True)
    
    parser.add_argument("--search_mode", type=str, default="greedy",
                        choices=["exhaustive", "greedy"])
    parser.add_argument("--test_num_batch", type=int, default=10)
    return parser


def wer_greedy_search_onestep(prior_clipped_layers,
                        remained_layers, model, dataloader):
    '''
    If at the starting point:
        prior_clipped_layers: []
        remained_layers: [1,2,3,4,5,6, 7, 8,9,10,11,12]
    '''
    # wer is the specified measure here
    measure_list = []
    for sel_layer in remained_layers:
        temp = copy.deepcopy(prior_clipped_layers)
        temp.append(sel_layer)
        temp = sorted(temp)
        regulated_temp = regulate_clip_layers_list(temp)

        # clipped model according the suggested layer pruning strategy
        new_config = model.config
        num_clip_layers = len(temp)
        new_config.num_hidden_layers = args.num_layers - num_clip_layers
        # logging.info("--- Construct the clipped model")
        clipped_model = ExtendedWav2Vec2ForCTC(config=new_config)
        # logging.info("--- Start copying weight from deepnet to shallownet")
        deepnet_statedict = model.state_dict()
        clipped_model.load_state_dict(deepnet_statedict, strict=False)
         
        # series_clipped_layers= [[2, 2], [4, 4], [6, 6]]
        # series_clipped_layers=[[2,4], [7,8], [11,11]]
        # series_clipped_layers=[[4, 6]]
        series_clipped_layers = copy.deepcopy(regulated_temp)
        # logging.info(f"series_clipped_layers: {series_clipped_layers}") 
         
        clip_num = 0
        for clip_layer in series_clipped_layers:
            clip_num += clip_layer[1] - clip_layer[0] +1
        assert clip_num == num_clip_layers
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
        # logging.info(f"-- WER stats on {len(Label_Trans)} utterances")
        clipped_wer = wer_metric.compute(predictions=Clipped_ASR_Trans,
                                        references=Label_Trans) 
        # logging.info(f"For {regulated_clip}: clipped wer : {clipped_wer*100}%") 
        measure_list.append(clipped_wer)

    # thwo lowest wer value will be selected
    top_idx = np.argsort(measure_list)[0]
    sel_layer = remained_layers[top_idx]
    prior_clipped_layers.append(sel_layer)
    prior_clipped_layers = sorted(prior_clipped_layers)  

    remained_layers.pop(top_idx)
    best_wer = measure_list[top_idx]
    # print("prior_clipped_layers: ", prior_clipped_layers)
    return prior_clipped_layers, remained_layers,  best_wer





if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    
    log_name = f"WER_greedy_clip{args.num_layers}to{args.num_layers - args.num_clip_layers}.log"
    setup_logger(log_name=log_name, save_dir='log')

    num_layers = args.num_layers
    layers = np.arange(1, num_layers+1)

   
    num_clip_layers = args.num_clip_layers

    logging.info(f"Layer Pruning {num_clip_layers} layers from {num_layers} layers")
    
    logging.info("--- Construct the dataset")
    # wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/wav.scp"
    # text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/text"
    text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/text"

    dataset = SpeechDataset(wav_scp=wav_scp, text_file=text_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=SpeechCollate())

    pretrained_path_name = "kehanlu/mandarin-wav2vec2-aishell1"
    config = Wav2Vec2Config.from_pretrained(pretrained_path_name)

    logging.info(f"-- construct the original {args.num_layers}-layer model")
    if args.num_layers == 12:
        assert config.num_hidden_layers == args.num_layers
        model =ExtendedWav2Vec2ForCTC(config=config)
        model = model.from_pretrained(pretrained_path_name)
    elif args.num_layers == 6:
        config.num_hidden_layers = args.num_layers
        model = ExtendedWav2Vec2ForCTC(config=config)
        pt_path = "../exp/clip12to6_ft_simple/asr_ep20.pt" 
        model.load_state_dict(torch.load(pt_path))
        logging.info(f"--load {pt_path} for 6 layer model")
    else:
        raise RuntimeError(f"Not supported {args.num_layers} at this time.")
    
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modle = model.to(device)
    processor = Wav2Vec2Processor.from_pretrained(pretrained_path_name)
    wer_metric = load_metric("wer")  
    
    
   
    if args.search_mode == "greedy":
        logging.info(f"Greedy search pruning ways")
        prior_clipped_layers = []
        remained_layers = []
        for i in range(1, num_layers+1):
            remained_layers.append(i)
        
        for i in range(1, num_clip_layers+1):
            prior_clipped_layers, remained_layers, best_wer = wer_greedy_search_onestep(
                prior_clipped_layers, remained_layers, model, dataloader)

            logging.info(f"prune {i} layer,  prior_clipped_layers: {regulate_clip_layers_list(prior_clipped_layers)}, \
                            best_wer: {best_wer*100}%")
    else:
        raise RuntimeError(f"Not supported such search_mode: {args.search_mode}")
    

    
   
