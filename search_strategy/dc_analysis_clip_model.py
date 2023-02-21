import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
import pdb
import sys
sys.path.append("../")

from data import SpeechDataset, SpeechCollate
from torch.utils.data import DataLoader

from DC import Distance_Correlation
from DC import distance_matrix, distance_correlation_from_distmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import argparse


from utils import relocate, str2bool, set_seed
import numpy as np

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", type=str2bool, default=True)

    parser.add_argument("--num_layers", type=int, default=12, required=True)
    parser.add_argument("--load_pt", type=str2bool, default=False, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("--- Construct the model")
    
    if args.num_layers == 12:
        print(f"--- use pretrained: {args.use_pretrained}")
        if args.use_pretrained:
            model = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
            # torch.save(model.state_dict(), 'kehanlu_hgface_asr.pt')
        else:
            config = Wav2Vec2Config.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
            model = ExtendedWav2Vec2ForCTC(config)

        model.config.output_hidden_states = True
        if args.load_pt:
            # load pretrained asr, continue train 50 epoch
            # pt_path = "asr_model_ft_aishell.pt"
            # pt_path = "exp/aishell_ft_bs64_lr1e-5/asr_ep2.pt"
            pt_path = "../exp/aishell_ft_bs64lr1e-5_freeze_cnn/asr_ep3.pt"
            model.load_state_dict(torch.load(pt_path))
            print(f"--- load {pt_path} pt done")

    elif args.num_layers == 6:
        config = Wav2Vec2Config.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        config.num_hidden_layers = args.num_layers
        config.output_hidden_states = True

        model = ExtendedWav2Vec2ForCTC(config)
        if args.load_pt:
            pt_path = "../exp/clip12to6_ft_simple/asr_ep20.pt"
            pt_path = "../exp/clip12to6_ft_simple/asr_avg_valid10.pt"
            model.load_state_dict(torch.load(pt_path))
            print(f"--- load {pt_path} pt done")
    else:
        raise RuntimeError(f"Not supported such num_layers: {args.num_layers} now")

    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    # audio_input, sample_rate = sf.read(
    #     "/lan/ibdata/SPEECH_DATABASE/aishell/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav")

    print("--- Construct the dataset")
    batch_size = 32
    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    dataset = SpeechDataset(wav_scp=wav_scp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, collate_fn=SpeechCollate())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sample_rate = 16000

    max_iter_num = 10

    for cnt, batch in enumerate(tqdm(dataloader)):
        if cnt >= max_iter_num:
            break

        audio_input = batch['audio'] 
        inputs = processor(audio_input, sampling_rate=sample_rate, 
                            padding=True, return_tensors="pt",
                            return_attention_mask=True,
                                dtype=torch.float32)
        inputs['input_values'] = inputs['input_values'].float()
        inputs = relocate(inputs, device)
        
        with torch.no_grad():
            model.eval()
            model_output = model(**inputs)
            # tuple of layer representations
            hidden_states = model_output.hidden_states
        
        num_layers = len(hidden_states)
        # print(f"--- iter {cnt} {num_layers} layers to be correlated")
        layers_distmat = []
        for layer_feat in hidden_states:
            # [B, T, D]
            # method 1:
            # layer_feat = layer_feat.reshape(batch_size, -1)
            # method 2:
            layer_feat = layer_feat.mean(dim=1)
            dist_mat = distance_matrix(layer_feat)
            layers_distmat.append(dist_mat)
        
        # pdb.set_trace()
        if cnt == 0:
            dc_mat = torch.zeros(num_layers, num_layers).to(device) 
        for i in range(num_layers):
            for j in range(num_layers):
                dc_mat[i][j] += distance_correlation_from_distmat(layers_distmat[i], layers_distmat[j])

    dc_mat = dc_mat / max_iter_num
    dc_mat = dc_mat.cpu().numpy()

    with open(f"dump/dc_mat_mean_{max_iter_num}_numlayer{args.num_layers}.npy", 'wb') as f:
        np.save(f, dc_mat)
        print("dc_mat py saves done")
 
    plt.figure()
    sns.heatmap(dc_mat, annot=True, vmin=0.5, fmt=".2f")
    # plt.title("[B, T*D]")
    # plt.savefig("./dc_mat.png")
    plt.title("[B, D]")
    # plt.savefig("./dc_mat_mean_10_rand.png")
    if args.load_pt:
        lab = pt_path.split("/")[-1].split('.')[0]
        plt.savefig(f"figure/dc_mat_mean_{max_iter_num}_{lab}_numlayer{args.num_layers}.png")
    else:
        plt.savefig(f"figure/dc_mat_mean_{max_iter_num}_numlayer{args.num_layers}.png")



