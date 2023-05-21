import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
import pdb

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
    parser.add_argument("--load_pt", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_layers", type=int, default=12)

    parser.add_argument("--batch_size", type=int, default=32, required=True)
    parser.add_argument("--accum_batch", type=int, default=1,
                    help="the function is similar to accumn_grad",
                    required=False)
    parser.add_argument("--iter_num", type=int, default=10, required=True)

    parser.add_argument("--mode", type=str, default="mean",
                        help="can be seq [B*T, D] or mean [B, D]")

    args = parser.parse_args()

    set_seed(args.seed)

    print("--- Construct the model")
    
    print(f"--- use pretrained: {args.use_pretrained}")
    if args.use_pretrained and args.num_layers == 12:
        model = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        # torch.save(model.state_dict(), 'kehanlu_hgface_asr.pt')
    elif args.num_layers == 6:
        config = Wav2Vec2Config.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        config.num_hidden_layers = args.num_layers
        model = ExtendedWav2Vec2ForCTC(config)
        model.load_state_dict(torch.load("exp/clip12to6/clip12to6.pt"))

    else:
        config = Wav2Vec2Config.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        model = ExtendedWav2Vec2ForCTC(config)
    
    model.config.output_hidden_states = True
    if args.load_pt:
        # load pretrained asr, continue train 50 epoch
        # pt_path = "asr_model_ft_aishell.pt"
        pt_path = "exp/aishell_ft_bs64_lr1e-5/asr_ep2.pt"
        model.load_state_dict(torch.load(pt_path))
        print(f"--- load {pt_path} pt done")
        

    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    # audio_input, sample_rate = sf.read(
    #     "/lan/ibdata/SPEECH_DATABASE/aishell/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav")

    print("--- Construct the dataset")
    batch_size = args.batch_size
    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    # wav_scp = "/home/louislau/research/prep_mfa/data/CSRC2021_RAW/SetC1/wav.scp"
    dataset = SpeechDataset(wav_scp=wav_scp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, collate_fn=SpeechCollate())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sample_rate = 16000

    max_iter_num = args.iter_num * args.accum_batch

    accum_hidden_states = []
    for i in range(args.num_layers+1):
        accum_hidden_states.append([])

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
        for i, layer_feat in enumerate(hidden_states):
            # [B, T, D]
            # method 1:
            # layer_feat = layer_feat.reshape(batch_size, -1)
            if args.mode == "mean":
                # method 2: [B, D]
                layer_feat = layer_feat.mean(dim=1)
            elif args.mode == "seq":
                # [B, D*T]
                B = layer_feat.shape[0]
                layer_feat = layer_feat.reshape(B, -1)
            accum_hidden_states[i].append(layer_feat)
        
        if cnt == 0:
            dc_mat = torch.zeros(num_layers, num_layers).to(device)
            # dc_mat = torch.zeros(num_layers, num_layers)
        
        if (cnt + 1) % args.accum_batch == 0:
            layers_distmat = []
            for layer_feat in accum_hidden_states:
                layer_feat = torch.cat(layer_feat, dim=0)            
    
                dist_mat = distance_matrix(layer_feat)
                layers_distmat.append(dist_mat)

                # pdb.set_trace()
            for i in range(num_layers):
                for j in range(num_layers):
                    dc_mat[i][j] += distance_correlation_from_distmat(layers_distmat[i], layers_distmat[j])
            
            accum_hidden_states = []
            for i in range(args.num_layers+1):
                accum_hidden_states.append([])

    dc_mat = dc_mat / args.iter_num
    dc_mat = dc_mat.cpu().numpy()

    if not args.load_pt:
        with open(f"dump/dc_mat_{args.mode}_bs{args.batch_size * args.accum_batch}_iter{args.iter_num}.npy", 'wb') as f:
            np.save(f, dc_mat)
            print("dc_mat py save done")
    else:
        lab = pt_path.split("/")[-1].split('.')[0]
        with open(f"dump/dc_mat_{args.mode}_bs{args.batch_size * args.accum_batch}_iter{args.iter_num}_{lab}.npy", 'wb') as f:
           np.save(f, dc_mat)
           print("dc_mat py save done")
    
    print(f"DC Mode: {args.mode}")
    print(f"In this DC measure analysis: Batch Size: {args.batch_size * args.accum_batch}")
    print(f"Iter number: {args.iter_num}")
    print(f"Total speech samples forward: {args.batch_size * args.accum_batch * args.iter_num}")

    plt.figure()
    # sns.heatmap(dc_mat, annot=True, vmin=0.5, fmt=".2f")
    sns.heatmap(dc_mat)
    # plt.title("[B, T*D]")
    # plt.savefig("./dc_mat.png")
    if args.mode == "mean":
        plt.title("[B, D]")
    elif args.mode == "seq":
        plt.title("[B, T*D]")
    # plt.savefig("./dc_mat_mean_10_rand.png")
    if args.load_pt:
        lab = pt_path.split("/")[-1].split('.')[0]
        plt.savefig(f"figure/dc_mat_{args.mode}_bs{args.batch_size * args.accum_batch}_iter{args.iter_num}_{lab}.png")
    else:
        plt.savefig(f"figure/dc_mat_{args.mode}_bs_{args.batch_size * args.accum_batch}_iter{args.iter_num}.png")


