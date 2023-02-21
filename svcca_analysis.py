import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
import pdb

from data import SpeechDataset, SpeechCollate
from torch.utils.data import DataLoader

from SVCCA import CCA, SVCCA, PWCCA

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
    
    parser.add_argument("--batch_size", type=int, default=32, required=True)
    parser.add_argument("--iter_num", type=int, default=10, required=True)
    
    parser.add_argument("--thre", type=float, default=0.99, 
                help="the thre percentage of singular values", required=True)
    parser.add_argument("--mode", type=str, default="T", 
                help="can be T (including time_seq as samples) or U (utterance-level samples)")
    args = parser.parse_args()

    set_seed(args.seed)

    print("--- Construct the model")
    
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
        pt_path = "exp/aishell_ft_bs64lr1e-5_freeze_cnn/asr_ep3.pt"
        model.load_state_dict(torch.load(pt_path))
        print(f"--- load {pt_path} pt done")
        

    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    # audio_input, sample_rate = sf.read(
    #     "/lan/ibdata/SPEECH_DATABASE/aishell/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav")

    print("--- Construct the dataset")
    batch_size = args.batch_size
    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    dataset = SpeechDataset(wav_scp=wav_scp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, collate_fn=SpeechCollate())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sample_rate = 16000

    max_iter_num = args.iter_num

    layers_neurons_activation = []
    for i in range(13):
        layers_neurons_activation.append([])

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
            if args.mode == "T":
                D = layer_feat.shape[-1]
                # [B*T, D]
                layer_feat = layer_feat.reshape(-1, D)
            elif args.mode == "U":
                layer_feat = torch.mean(layer_feat, dim=1)
            
            layers_neurons_activation[i].append(layer_feat.cpu())
        
    for i in range(len(layers_neurons_activation)):
        temp = torch.cat(layers_neurons_activation[i], dim=0)
        # [D, N_samples]
        temp = temp.transpose(0,1).contiguous().numpy()
        layers_neurons_activation[i] = temp

    
    print(f"The running mode: {args.mode}")
    print(f"The NUM_Neurons IS EQUAL TO {layers_neurons_activation[0].shape[0]}")
    print(f"The NUM_SAMPLES IS EQUAL TO {layers_neurons_activation[0].shape[-1]}")
    print(f"--- sample/neuron ratio is {layers_neurons_activation[0].shape[-1]/layers_neurons_activation[0].shape[0]}")
    print(f"Total speech samples: {args.iter_num * args.batch_size}")

    cca_mat = np.zeros([num_layers, num_layers])
    for i in range(0, num_layers):
        for j in range(i, num_layers):
            print(f"--- Processing [{i} ,{j}]")
            cca_mat[i][j] = np.mean(SVCCA(
                layers_neurons_activation[i], layers_neurons_activation[j], thre=args.thre))
    # cca_mat = cca_mat + cca_mat.T - np.diag(cca_mat)
    cca_mat = cca_mat + cca_mat.T - np.eye(len(cca_mat))

    with open(f"dump/cca_mat_{args.iter_num * args.batch_size}{args.mode}_svthre{args.thre}.npy", 'wb') as f:
        np.save(f, cca_mat)
        print("cca_mat py save done")
 
    plt.figure()
    # sns.heatmap(dc_mat, annot=True, vmin=0.5, fmt=".2f")
    sns.heatmap(cca_mat)
    # plt.title("[B, T*D]")
    # plt.savefig("./dc_mat.png")
    plt.title("SVCCA")
    # plt.savefig("./dc_mat_mean_10_rand.png")
    if args.load_pt:
        lab = pt_path.split("/")[-1].split('.')[0]
        plt.savefig(f"./cca_mat_{lab}_freeze_cnn.png")
    else:
        # plt.savefig("./dc_mat_mean_100_annot.png")
        plt.savefig(f"figure/cca_mat_{args.iter_num * args.batch_size}{args.mode}_svthre{args.thre}.png")


