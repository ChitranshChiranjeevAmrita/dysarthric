
import numpy as np
import torch
from utils import utils


class XvecSpeechGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode, args):
        """
        Read the textfile and get the paths
        """
        self.mode = mode
        self.win_length = args.win_length
        self.n_fft = args.n_fft
        self.audio_links = []
        self.labels = []
        LABELS = {
            "HI": 3, "MI": 2, "LO": 1, "VL": 0, "NO": -1
        }

        with open(manifest, 'r') as file:
            for idx, line in enumerate(file):
                parts = line.rstrip('\n').split(',')
                try:
                    self.audio_links.append(parts[0])
                    self.labels.append(int(parts[1]))
                except Exception as e:
                    print(line)



    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        win_length = self.win_length
        n_fft = self.n_fft
        # lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        spec = np.load(audio_link, allow_pickle=True)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id)),
                  'path': audio_link
                  }
        return sample

