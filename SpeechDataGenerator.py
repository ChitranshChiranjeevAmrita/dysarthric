
import numpy as np
import torch
from utils import utils

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode, args):
        """
        Read the textfile and get the paths
        """
        self.mode=mode
        self.win_length = args.win_length
        self.n_fft = args.n_fft
        self.audio_links = []
        self.labels = []
        LABELS = {
            "HI": 3, "MI":2, "LO":1, "VL":0, "NO": -1
        }

        with open(manifest, 'r') as file:
            for idx, line in enumerate(file):
                parts = line.rstrip('\n').split(',')
                if idx != 0 and len(parts) > 1:
                    self.audio_links.append(parts[3])
                    self.labels.append(LABELS[parts[4]])
        #self.labels = [idx, line.rstrip('\n').split(',')[4] for idx, line in enumerate(open(manifest)) if idx != 0 and mode != 'test']
        
        if mode == 'test':
           for i in range(len(self.audio_links)):
           	self.labels.append(1)
        

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]
        class_id = self.labels[idx]
        win_length = self.win_length
        n_fft = self.n_fft
        #lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        #spec = utils.load_data(audio_link, win_length=win_length, n_fft=n_fft, mode=self.mode)
        spec = utils.feature_extraction(audio_link, sr=16000, min_dur_sec=1, win_length=400, hop_length=80, n_mfcc=40, n_mels=40, spec_len=400,mode='train')
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id)),
                  'path': audio_link 
                }
        return sample

