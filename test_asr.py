import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model
import pdb

from data import SpeechDataset, SpeechCollate
from torch.utils.data import DataLoader, Dataset


from transformers import Trainer, TrainingArguments

from tqdm import tqdm
from utils import set_seed, relocate, str2bool

from torch.optim import AdamW, Adam

from torch.optim.lr_scheduler import LambdaLR

from train import ASRDataset, ASRCollate, ExtendedWav2Vec2ForCTC


        
import argparse
def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="aishell_ft", required=True)
    parser.add_argument("--resume_epoch", type=int, default=0, required=True)


    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--pretrained_path_name", type=str,
                        default="kehanlu/mandarin-wav2vec2-aishell1",
                        # default="qinyue/wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1"
                    )
    parser.add_argument("--use_pretrained", type=str2bool, default=True)

    parser.add_argument("--test_num_batch", type=int, default=50)
    return parser



from datasets import load_metric
import time
import os
from data import SpeechDataset, SpeechCollate
from utils import setup_logger
import logging



if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = setup_parser()
    args = parser.parse_args()

    args.output_dir = f"exp/{args.output_dir}"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    setup_logger(log_name="test_asr.log",save_dir=args.output_dir)
    logging.info(f"output_dir: {args.output_dir}")

    set_seed(seed=args.seed)

    # test_wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/wav.scp"
    test_wav_scp = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_dev/wav.scp"
    # test_text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_test/text"
    test_text_file = "/home/louislau/research/prep_mfa/data/AISHELL1/aishell_dev/text"
    
    test_dataset = SpeechDataset(wav_scp=test_wav_scp, text_file=test_text_file)
    print("--- test dataset constructed done")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=SpeechCollate())
    processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

    print("--- Construct the model")
    model_orig = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
    new_config = model_orig.config

    logging.info(f"ASR model has: {args.num_layers} layers")
    new_config.num_hidden_layers = args.num_layers

    model = ExtendedWav2Vec2ForCTC(config=new_config) 

    
    # model size
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Num of model params: {num_params / 1e6}M")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() 
    # wer_metric = load_metric("wer")  
    wer_metric = load_metric("cer")  
    
    if args.use_pretrained and args.num_layers == 12:
        # ckpt = "./kehanlu_hgface_asr.pt"
        # model.load_state_dict(torch.load(ckpt))
        model = model.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
        model.to(device)
        logging.info(f"Loading model ckpt from pretrained hgface model")
    if args.resume_epoch > 0:
        ckpt = os.path.join(args.output_dir, f'asr_ep{args.resume_epoch}.pt')
        ckpt = os.path.join(args.output_dir, f'asr_avg_valid10.pt')
        model.load_state_dict(torch.load(ckpt))
        logging.info(f"Loading model ckpt from {ckpt}")

    model.eval()
    ASR_Trans = []
    Label_Trans = [] 

    sample_rate = 16000
    
    # real time factor
    decode_time = 0
    overall_process_time = 0
    utterance_time = 0

    overall_start_time = time.time()
    for i, batch in enumerate(tqdm(test_dataloader)):
        # if i >= args.test_num_batch:
        #     break
        audio_input = batch['audio'] 
        inputs = processor(audio_input, sampling_rate=sample_rate, 
                            padding=True, return_tensors="pt",
                            return_attention_mask=True,
                                # dtype=torch.float32
                            )
        inputs['input_values'] = inputs['input_values'].float()
        inputs = relocate(inputs, device)
        with torch.no_grad():
            start_time =time.time()
            model_output = model(**inputs)
            logits = model_output.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            decode_time += (time.time() - start_time)
            # utterance_time += torch.sum(inputs['attention_mask']) / sample_rate
            B, T = inputs["input_values"].shape
            utterance_time += B*T / sample_rate
            label_transcription = batch['text']

            # seems wer here need to separated by space
            # pdb.set_trace()
            # transcription = [" ".join(t) for t in transcription]
            # label_transcription = [" ".join(t) for t in label_transcription]

            ASR_Trans += transcription
            Label_Trans += label_transcription
    
    overall_process_time = time.time() - overall_start_time
    # print(f"Decoding Elapsed {time.time() - start_time}")
    wer = wer_metric.compute(predictions=ASR_Trans,
                                    references=Label_Trans) 
    # print(f"----EPOCH: {epoch}, Test wer : {wer*100}%")
    logging.info(f"On {len(Label_Trans)} Utts, Test wer : {wer*100}%")

    # real time factor, decoding time ratio:
    rtf = decode_time / utterance_time 
    logging.info(f"Test decoding RTF: {rtf}")
    # rtf2 = overall_process_time / utterance_time
    # logging.info(f"Test overall process RTF: {rtf2}")
        
        