import torch
import os
import pdb

from torch.utils.data import Dataset, DataLoader
import soundfile as sf


class SpeechDataset(Dataset):
    ''' Read a list of audio files
    '''
    def __init__(self, wav_scp, text_file=None):
        super().__init__()

        self.wav_scp =  wav_scp
        self.utt2wavpath = self._read_wav_scp(wav_scp)
        self.uttids = list(self.utt2wavpath.keys())

        self.text_file = text_file
        if self.text_file is not None:
            self.utt2text = self._read_text(text_file)
        

    def _read_wav_scp(self, wav_scp):
        utt2wavpath = {}
        with open(wav_scp, 'r') as fd:
            for line in fd:
                uttid, wavpath = line.strip().split()
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
        y, sr = sf.read(self.utt2wavpath[uttid])
        if self.text_file is not None:
            text = self.utt2text[uttid]
            text = "".join(text.split())
            return {'audio': y, 'sr': sr, 'uttid': uttid, 'text': text}
        else:
            return {'audio': y, 'sr': sr, 'uttid': uttid}


class SpeechCollate():
    def __call__(self, batch):
        audio_list = []
        uttid_list = []
        text_list = []

        for item in batch:
            audio_list.append(item['audio'])
            uttid_list.append(item['uttid'])
            if 'text' in item:
                text_list.append(item['text'])
        if 'text' in batch[0]:
            return {'audio': audio_list, \
                'uttid': uttid_list, 'text': text_list}
        else:
            return {'audio': audio_list, 'uttid': uttid_list}







if __name__ == "__main__":
    wav_scp = "/home/userxx/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    dataset = SpeechDataset(wav_scp=wav_scp)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=4, collate_fn=SpeechCollate())
    for batch in dataloader:
        pdb.set_trace()
