import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
from transformers import Wav2Vec2Model
import pdb

from torch.utils.data import DataLoader, Dataset


from transformers import Trainer, TrainingArguments

from tqdm import tqdm
from utils import set_seed, relocate, str2bool

from torch.optim import AdamW, Adam
import argparse

class ASRDataset(Dataset):
    '''
    Datset for ASR training
    '''
    def __init__(self, wav_scp, text_file):
        super().__init__()
        self.utt2wavpath = self._read_wavscp(wav_scp)
        self.utt2text = self._read_text(text_file)

        self.uttids = list(self.utt2wavpath.keys())

    def _read_wavscp(self, wav_scp):
        utt2wavpath = {}
        with open(wav_scp, "r") as fd:
            for line in fd:
                uttid, wavpath = line.strip().split(maxsplit=1)
                utt2wavpath[uttid] = wavpath
        return utt2wavpath
    
    def _read_text(self, text_file):
        utt2text = {}
        with open(text_file, "r") as fd:
            for line in fd:
                uttid, text = line.strip().split(maxsplit=1)
                utt2text[uttid] = text
        return utt2text

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        wavpath = self.utt2wavpath[uttid]
        text = self.utt2text[uttid]
        # text: a b c -> abc, no space betweeen chinese characters
        text = ''.join(text.split())
        speech, _ = sf.read(wavpath)
        return {'uttid': uttid, 'speech': speech, 'text': text}


class ASRCollate():
    def __init__(self, w2v2_processor_name="kehanlu/mandarin-wav2vec2-aishell1"):
        self.processor = Wav2Vec2Processor.from_pretrained(
            w2v2_processor_name
        )
        self.sample_rate = 16000

    def __call__(self, batch):
        new_batch = {'input_values': [], 'labels': [], 'uttids': []}
        for item in batch:
            new_batch['input_values'].append(item['speech'])
            new_batch['labels'].append(item['text'])
            new_batch['uttids'].append(item['uttid'])
        
        inputs = self.processor(new_batch['input_values'],
                                        sampling_rate=self.sample_rate, 
                                        padding=True, return_tensors="pt",
                                        return_attention_mask=True,
                                        dtype=torch.float32)
        # Two keys: input_values and attention_mask [batch_size, seq_len]
        new_batch.update(inputs)
        new_batch['input_values'] = new_batch['input_values'].float()

        with self.processor.as_target_processor():
            labels_batch = self.processor(new_batch['labels'],
                            padding=True,
                            return_tensors="pt")
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        new_batch['labels'] = labels

        return new_batch


        
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

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="clip12to6_ft", required=True)
    parser.add_argument("--resume_epoch", type=int, default=0, required=True)

    parser.add_argument("--num_layers", type=int, default=12, required=True)
    parser.add_argument("--num_clip_layers", type=int, default=6, required=True)

    parser.add_argument("--clip_mode", type=str, default="simple_copy", required=False,
                    choices=["scratch", "simple_copy", "copy_adapter"])
    
    parser.add_argument("--distill_mode", type=str, default="kl_distll",
                    choices=["kl_distill", "l2_distll", "connect_l2_distll"],
                    help="Temporaily not to be used in fine-tuning, simple ft is applied by default.")

    parser.add_argument("--test_num_batch", type=int, default=10)

    parser.add_argument("--lr", type=float, default=2e-5,
                            help="set initial learning rate")

    parser.add_argument("--max_grad_norm", type=float, default=-1)
    parser.add_argument("--layerdrop", type=float, default=0.1, required=False)
    parser.add_argument("--use_scheduler", type=str2bool, default=False, required=False)
    return parser



from datasets import load_metric
import time
import os
from data import SpeechDataset, SpeechCollate
from utils import setup_logger
import logging

from copy_clipped_model import downlayer_copy_weight_series

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    args.output_dir = f"exp/{args.output_dir}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    setup_logger(log_name="train.log",save_dir=args.output_dir)

    set_seed(seed=args.seed)

    wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_train/text"    
    asr_dataset = ASRDataset(wav_scp, text_file)
    
    print("-- asr dataset constructed done.")
    asr_dataloader = DataLoader(asr_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, drop_last=False, collate_fn=ASRCollate())

    
    test_wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/wav.scp"
    test_text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/text"
    test_dataset = SpeechDataset(wav_scp=test_wav_scp, text_file=test_text_file)

    print("--- test dataset constructed done")
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=SpeechCollate())
    
    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    print("--- Construct the original model")
    if args.num_layers == 12:
        model_orig = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        # model_orig.config.output_hidden_states = True
    elif args.num_layers == 6:
        config = Wav2Vec2Config.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        config.num_hidden_layers = args.num_layers
        model_orig = ExtendedWav2Vec2ForCTC(config=config)
        pt_path = "exp/clip12to6_ft_simple/asr_ep20.pt"
        pt_path = "exp/clip12to6_ft_simple/asr_ep48.pt"
        model_orig.load_state_dict(torch.load(pt_path))
        # model_orig.config.output_hidden_states = True
    else:
        raise RuntimeError(f"not supported such num_layers: {args.num_layers}")
    
    print(f"-- CLIP mode: {args.clip_mode}")
    if args.clip_mode == "simple_copy":
        '''
        Put the pruning strategy here, as a starting point to fine-tune.
        '''
        
        # series_clipped_layers=[[2,4], [7,9]]
        # series_clipped_layers=[[2,4], [7,8], [11,11]]
        series_clipped_layers = [[1,1], [3,3], [5,5], [7,7], [9,9], [11,11]]
        # series_clipped_layers = [[2,2], [4,4], [6,6], [8,8], [10,10], [12,12]]
        # series_clipped_layers=[[1, 6]]
        # series_clipped_layers=[[2, 7]]
        # series_clipped_layers=[[3, 8]]
        # series_clipped_layers = [[1,3], [5,7], [9,11]] 
        # series_clipped_layers = [[1,1], [3,3], [5,5]] 
        # series_clipped_layers = [[4,4], [7,7], [10,10]] 
        # greedy 
        series_clipped_layers = [[1, 1], [4, 5], [7, 7], [10, 10], [12, 12]]
        # dc find clip12to6
        series_clipped_layers = [[2, 3], [5, 5], [7, 7], [9, 9], [11, 11]]
        # svcca find: 11.08% wer lp:
        series_clipped_layers = [[1, 1], [3, 3], [5, 5], [7, 7], [10, 10], [12, 12]]
        
        # series_clipped_layers=[[3, 3]]
        # series_clipped_layers=[[4, 9]]
        # series_clipped_layers = [[2, 2], [4, 4], [8, 8]]
        # series_clipped_layers=[[5, 10]]
        # series_clipped_layers=[[6, 11]]
        # series_clipped_layers=[[7, 12]]
        # series_clipped_layers=[[1, 6]]
        
        
        # dc bs32 best: lp_wer: 15.36
        # series_clipped_layers = [[1, 1], [3, 4], [7, 7], [9, 9], [11, 11]]

        # dc bs32 best wer: lp_wer: 14.56
        # series_clipped_layers = [[2, 4], [7, 7], [9, 9], [11, 11]]
        # dc bs32 dc best2: lp_wer: 44.43
        # series_clipped_layers = [[1,2], [4,4], [7,7], [9,9], [11,11]]
        
        new_config = model_orig.config
        clipped_layers = 0
        for l in series_clipped_layers:
            clipped_layers += (l[1] - l[0] + 1)
        logging.info(f"--- CLIP {clipped_layers} layers")
        logging.info(f"--- Series Clipped layers: {series_clipped_layers}")

        assert clipped_layers == args.num_clip_layers

        new_config.num_hidden_layers = args.num_layers - clipped_layers
        # no conect_layers attribute means no added modules between layers
        # new_config.update({'connect_layers': [0, 2]})

        print("--- Construct the clipped model")
        model = ExtendedWav2Vec2ForCTC(config=new_config)

        print("--- Start copying weight from deepnet to shallownet")
        deepnet_statedict = model_orig.state_dict()
        model.load_state_dict(deepnet_statedict, strict=False)
        shallownet_statedict = model.state_dict()
        shallownet_statedict =  downlayer_copy_weight_series(
            deepnet_statedict, shallownet_statedict,
            series_clipped_layers=series_clipped_layers
        )
        model.load_state_dict(shallownet_statedict, strict=True)

        # torch.save(model.state_dict(), os.path.join(args.output_dir, 'clip12to6.pt'))
        # pdb.set_trace()

    elif args.clip_mode == "scratch":
        new_config = model_orig.config
        clipped_layers = args.num_clip_layers
        logging.info(f"--- CLIP {clipped_layers} layers")
        new_config.num_hidden_layers = args.num_layers - clipped_layers
        print("--- Construct the clipped model")
        model = ExtendedWav2Vec2ForCTC(config=new_config)

    else:
        raise RuntimeError(f"Not Support such Clip_mode: {args.clip_mode}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig.to(device)
    model_orig.train() 
    
    model.to(device)
    model.train()

    # added 06/12/2022
    if model.config.num_hidden_layers == 3:
        # already very shallow layers, no need to regularize again.
        logging.info("model config layerdrop set to be 0.0")
        model.config.layerdrop = 0.0

    logging.info(f"Layerdrop of the clipped model: {args.layerdrop}")
    model.config.layerdrop = args.layerdrop

    if not args.clip_mode == "scratch":
        # logging.info("-- model freeze feature extractor set to be True")
        # model.freeze_feature_extractor()
        pass
    
    max_epoch = 50

    logging.info(f"--Initial learning rate: {args.lr}")
    # optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    optim = AdamW(model.parameters(), lr=args.lr)
    # lr_scheduler = get_linear_schedule_with_warmup(optim,
    #                     num_warmup_steps=30000, num_training_steps=len(asr_dataloader)*max_epoch)

    # milestones1 = [20, 40, 50]
    # milestones2 = [10, 40, 50]
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optim, milestones=milestones2, gamma=0.5
    # )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=10)
    print(f"--num_training_steps: {len(asr_dataloader)*max_epoch}")
    wer_metric = load_metric("wer")  
    
    if args.resume_epoch > 0:
        ckpt = os.path.join(args.output_dir, f'asr_ep{args.resume_epoch}.pt')
        model.load_state_dict(torch.load(ckpt))
        print(f"Loading model ckpt from {args.resume_epoch} epoch")

    for epoch in range(args.resume_epoch+1, max_epoch+1):
        if epoch == args.resume_epoch + 1:
            logging.info("Testing on 320 utterances to check the WER ---") 
            model.eval()
            ASR_Trans = []
            Label_Trans = [] 

            sample_rate = 16000
            start_time = time.time()
            for i, batch in enumerate(tqdm(test_dataloader)):
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
                    model_output = model(**inputs)
                    logits = model_output.logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    label_transcription = batch['text']

                    # seems wer here need to separated by space
                    transcription = [" ".join(t) for t in transcription]
                    label_transcription = [" ".join(t) for t in label_transcription]

                    ASR_Trans += transcription
                    Label_Trans += label_transcription
            # print(f"Decoding Elapsed {time.time() - start_time}")
            print(f"-- WER stats on {len(Label_Trans)} utterances")
            wer = wer_metric.compute(predictions=ASR_Trans,
                                            references=Label_Trans) 
            # print(f"----EPOCH: {epoch}, Test wer : {wer*100}%")
            logging.info(f"----EPOCH: {epoch-1}, Test wer : {wer*100}%")
        
        
        print(f'--- Epoch {epoch} asr training')
        model.train()
        avg_loss = 0
        for i, batch in enumerate(tqdm(asr_dataloader)):
            batch.pop('uttids')
            batch = relocate(batch, device)
            # 3 keys: loss, logits, hidden_states
            model_output = model(**batch)
            loss = model_output.loss
            avg_loss += loss.item()

            # normalzie the loss
            loss = loss / args.grad_accum 
            loss.backward()
            if args.max_grad_norm != -1:
                # grad norm clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            
            if ((i+1) % args.grad_accum == 0) or (i+1) == len(asr_dataloader):
                optim.step()
                # lr_scheduler.step()

                optim.zero_grad()
        
       
        avg_loss /= (i+1)
        logging.info(f'--- Epoch {epoch} asr training avg loss: {avg_loss:.2f}')
        # pdb.set_trace()
        torch.save(model.state_dict(), f'{args.output_dir}/asr_ep{epoch}.pt')
        # rm_asr_pt = f"{args.output_dir}/asr_ep{epoch-5}.pt"
        # if os.path.exists(rm_asr_pt):
        #     os.system(f"rm {rm_asr_pt}")
        
        if (epoch + 1) % 1 == 0:
            logging.info("Testing on 320 utterances to check the WER ---") 
            model.eval()
            ASR_Trans = []
            Label_Trans = [] 

            sample_rate = 16000
            start_time = time.time()
            for i, batch in enumerate(tqdm(test_dataloader)):
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
                    model_output = model(**inputs)
                    logits = model_output.logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    label_transcription = batch['text']

                    # seems wer here need to separated by space
                    transcription = [" ".join(t) for t in transcription]
                    label_transcription = [" ".join(t) for t in label_transcription]

                    ASR_Trans += transcription
                    Label_Trans += label_transcription
            # print(f"Decoding Elapsed {time.time() - start_time}")
            print(f"-- WER stats on {len(Label_Trans)} utterances")
            wer = wer_metric.compute(predictions=ASR_Trans,
                                            references=Label_Trans) 
            # print(f"----EPOCH: {epoch}, Test wer : {wer*100}%")
            logging.info(f"----EPOCH: {epoch}, Test wer : {wer*100}%")
        
        if args.use_scheduler:
            # epoch level change lr
            lr_scheduler.step(wer)
        
            




    