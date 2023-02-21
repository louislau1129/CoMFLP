import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model
import pdb

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

# W2V2_PROCESSOR = Wav2Vec2Processor.from_pretrained(
#            "kehanlu/mandarin-wav2vec2-aishell1"
#       )



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
    parser.add_argument("--test_num_batch", type=int, default=10)
    parser.add_argument("--load_pt", type=str2bool, default=False)

    return parser



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()


    set_seed(args.seed)
    print("--- Construct the original model")
    a = [1,2]
    net = nn.ModuleList([])
    for i in range(2):
        net.append(nn.Linear(256,256))
    
    model = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
    model.config.output_hidden_states = True
    
    if args.load_pt:
        # load pretrained asr, continue train 50 epoch
        pt_path = "exp/aishell_ft_bs64_lr1e-5/asr_ep7.pt"
        # pt_path = "exp/clip24to12_simple_ft/asr_ep19.pt"
        model.load_state_dict(torch.load(pt_path, map_location='cpu'))
        print("--- load pt done")
    
    new_config = model.config
    new_config.num_hidden_layers = 12 - 6
    # new_config.update({'connect_layers': [0, 2]})

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
    
    # series_clipped_layers=[[2,4], [7,9]]
    # series_clipped_layers=[[2,4], [7,8], [11,11]]
    series_clipped_layers = [[1,1], [3,3], [5,5], [7,7], [9,9], [11,11]]
    # series_clipped_layers = [[2,2], [4,4], [6,6], [8,8], [10,10], [12,12]]
    # series_clipped_layers=[[1, 6]]
    # series_clipped_layers=[[2, 7]]
    # series_clipped_layers=[[3, 8]]
    # series_clipped_layers=[[3, 3]]
    # series_clipped_layers=[[4, 9]]
    # series_clipped_layers=[[5, 10]]
    # series_clipped_layers=[[6, 11]]
    # series_clipped_layers=[[7, 12]]

    shallownet_statedict = clipped_model.state_dict()
    shallownet_statedict =  downlayer_copy_weight_series(
        deepnet_statedict, shallownet_statedict,
        series_clipped_layers=series_clipped_layers
    )
    clipped_model.load_state_dict(shallownet_statedict, strict=True)
    # print(f"Selected copying weight elapsed {time.time() - start_time}")
    
    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    # audio_input, sample_rate = sf.read(
    #     "/lan/ibdata/SPEECH_DATABASE/aishell/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav")

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
        with torch.no_grad():
            clipped_model_output = clipped_model(**inputs)
            model_output = model(**inputs)
            clipped_logits = clipped_model_output.logits
            logits = model_output.logits
            # hidden_states = model_output.hidden_states
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

            '''
            # compute wer over this batch list
            wer = wer_metric.compute(predictions=transcription,
                                    references=label_transcription) 
            clipped_wer = wer_metric.compute(predictions=clipped_transcription,
                                    references=label_transcription) 

            print(f"wer : {wer*100}%")   
            print(f"clipped wer : {clipped_wer*100}%")   
            pdb.set_trace()
            
        for t, clipped_t in zip(transcription, clipped_transcription):
            print(f'asr: {t}')
            print(f'clipped asr: {clipped_t}')
        pdb.set_trace()
        '''
    print(f"Decoding Elapsed {time.time() - start_time}")
    print(f"-- WER stats on {len(Label_Trans)} utterances")
    wer = wer_metric.compute(predictions=ASR_Trans,
                                    references=Label_Trans) 
    clipped_wer = wer_metric.compute(predictions=Clipped_ASR_Trans,
                                    references=Label_Trans) 

    print(f"wer : {wer*100}%")
    print(f"series_clipped_layers: {series_clipped_layers}")  
    print(f"clipped wer : {clipped_wer*100}%")   

    # for k,v in clipped_model.named_parameters():
    #     if k.startswith("wav2vec2.encoder.layers"):



    print("done")

    


