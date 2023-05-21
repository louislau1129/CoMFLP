import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model
import pdb
import sys
sys.path.append("../")


from data import SpeechDataset, SpeechCollate
from torch.utils.data import DataLoader, Dataset


from transformers import Trainer, TrainingArguments

from tqdm import tqdm
from utils import set_seed, relocate

from torch.optim import AdamW, Adam
import argparse
import os

import numpy as np
import time
from utils import set_seed, str2bool

import torch.nn as nn

from datasets import load_metric
import argparse

class ExtendedWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    In ESPNET there is a LayerNorm layer between encoder output and CTC classification head.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = torch.nn.Sequential(
                torch.nn.LayerNorm(config.hidden_size),
                self.lm_head
        )



def downlayer_copy_weight(deepnet_statedict, 
                        shallownet_statedict,
                        clip_layer=[2,4],
                        already_clip=0):
    '''
    only consider the wav2vec2.encoder.layers.{which_layer}
    Before call this func
    we assume you have load_state_dict(deepnet_statedict, strict=False)
    for shallow_model
    '''
    prefix = "wav2vec2.encoder.layers"
    num_clipped_layers = clip_layer[1] - clip_layer[0] + 1
    clip_layer[0], clip_layer[1] = \
            clip_layer[0] - already_clip, clip_layer[1] - already_clip
    
    for k, v in shallownet_statedict.items():
        if k.startswith(prefix) and 'adapterlayer' not in k and \
            "prior_transformlayer" not in k:
            i = int(k.split('.')[3])
            if i >= clip_layer[0]-1:
                deep_l = i + already_clip + num_clipped_layers
                deep_k = f"{prefix}.{deep_l}.{'.'.join(k.split('.')[4:])}"
                shallownet_statedict[k] = deepnet_statedict[deep_k]
    return shallownet_statedict
    

def downlayer_copy_weight_series(
    deepnet_statedict,
    shallownet_statedict,
    series_clipped_layers=[[2,4], [7,9]]):

    already_clip = 0
    for clip_layer in series_clipped_layers:
        shallownet_statedict = downlayer_copy_weight(
                        deepnet_statedict,
                        shallownet_statedict,
                        clip_layer,
                        already_clip=already_clip)
        already_clip += clip_layer[1] - clip_layer[0] + 1
    return shallownet_statedict
        


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_layers", type=int, default=12, required=True)
    parser.add_argument("--num_clip_layers", type=int, default=6, required=True)
    parser.add_argument("--test_num_batch", type=int, default=10)
    parser.add_argument("--load_pt", type=str2bool, default=False)

    return parser



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()


    set_seed(args.seed)
    
    model = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    if args.num_layers == 6:
        # 5.2% wer, 19.8 ctc loss
        pt_path = "../exp/clip12to6_ft_simple/asr_ep20.pt" 
        print(f"--- load {pt_path}")
        config = model.config
        config.num_hidden_layers = args.num_layers
        model = ExtendedWav2Vec2ForCTC(config=config)
        model.load_state_dict(torch.load(pt_path))

    if args.load_pt:
        # load pretrained asr, continue train 50 epoch
        # pt_path = "exp/aishell_ft_bs64_lr1e-5/asr_ep7.pt"
        pt_path = "exp/aishell_ft_bs64lr1e-5_nolayerdrop/asr_ep1.pt"
        pt_path = "exp/aishell_ft_bs64lr1e-5_freeze_cnn/asr_ep1.pt"
        pt_path = "../exp/ft_12_layerdrop0.5/asr_ep30.pt"
        model.load_state_dict(torch.load(pt_path))
        print("--- load pt done")
    
    new_config = model.config
    new_config.num_hidden_layers = args.num_layers - args.num_clip_layers

    print("--- Construct the clipped model")
    clipped_model = ExtendedWav2Vec2ForCTC(config=new_config)
    
    print("--- Start copying weight from deepnet to shallownet")
    start_time = time.time()
    deepnet_statedict = model.state_dict()
    clipped_model.load_state_dict(deepnet_statedict, strict=False)

    '''
    shallownet_statedict = downlayer_copy_weight(
        deepnet_statedict, clipped_model.state_dict(),
        clip_layer=[2,4])
    clipped_model.load_state_dict(shallownet_statedict)
    
    shallownet_statedict = downlayer_copy_weight(
        deepnet_statedict, clipped_model.state_dict(),
        clip_layer=[7,9], already_clip=3)
    clipped_model.load_state_dict(shallownet_statedict)
    '''
    
    series_clipped_layers=[[2,4], [7,9]]
    series_clipped_layers= [[2, 2], [4, 4], [6, 6]]
    # series_clipped_layers=[[2,4], [7,8], [11,11]]
    series_clipped_layers = [[1,1], [3,3], [5,5], [7,7], [9,9], [11,11]]
    # series_clipped_layers = [[1, 1], [3, 3], [5, 11]]
    # series_clipped_layers = [[2,2], [4,4], [6,6], [8,8], [10,10], [12,12]]
    # series_clipped_layers = [[2,4], [6,8], [10,12]]
    # series_clipped_layers = [[1, 2], [4, 4], [7, 7], [9, 9], [11, 11]]
    # series_clipped_layers=[[1, 6]]
    # series_clipped_layers=[[2, 7]]
    # series_clipped_layers=[[3, 8]]
    # series_clipped_layers=[[3, 3]]
    # series_clipped_layers=[[4, 9]]
    # series_clipped_layers=[[5, 10]]
    # series_clipped_layers=[[6, 11]]
    # series_clipped_layers=[[7, 12]]
    # series_clipped_layers=[[4, 6]]
    # series_clipped_layers = [[2, 2], [4, 5], [7, 7], [9, 9], [11, 11]]

    print(f"series_clipped_layers: {series_clipped_layers}") 
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
    # print(f"Selected copying weight elapsed {time.time() - start_time}")
    
    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")


    print("--- Construct the dataset")
    # wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/wav.scp"
    # text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/text"
    text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/text"

    dataset = SpeechDataset(wav_scp=wav_scp, text_file=text_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=SpeechCollate())
    clipped_model.eval()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    clipped_model.to(device)

    wer_metric = load_metric("wer")  
    ASR_Trans = []
    Clipped_ASR_Trans = []
    Label_Trans = [] 

    sample_rate = 16000
    start_time = time.time()
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
            logits = model_output.logits
            clipped_predicted_ids = torch.argmax(clipped_logits, dim=-1)
            predicted_ids = torch.argmax(logits, dim=-1)
            clipped_transcription = processor.batch_decode(clipped_predicted_ids)
            transcription = processor.batch_decode(predicted_ids)
            label_transcription = batch['text']
            
            # seems wer here need to separated by space
            transcription = [" ".join(t) for t in transcription]
            clipped_transcription = [" ".join(t) for t in clipped_transcription]
            label_transcription = [" ".join(t) for t in label_transcription]

            ASR_Trans += transcription
            Clipped_ASR_Trans += clipped_transcription
            Label_Trans += label_transcription

    print(f"Decoding Elapsed {time.time() - start_time}")
    print(f"-- WER stats on {len(Label_Trans)} utterances")
    wer = wer_metric.compute(predictions=ASR_Trans,
                                    references=Label_Trans) 
    clipped_wer = wer_metric.compute(predictions=Clipped_ASR_Trans,
                                    references=Label_Trans) 

    print(f"wer : {wer*100}%")
    print(f"clipped wer : {clipped_wer*100}%")   

    


