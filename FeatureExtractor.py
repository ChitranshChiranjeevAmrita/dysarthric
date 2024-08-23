import torch
from torch.utils.data import DataLoader
from SpeechDataGenerator import SpeechDataGenerator
import os
import numpy as np
import argparse
from utils.utils import speech_collate
from utils.utils import writeFeatures

torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import shutil

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-development_filepath', type=str, default='meta/development_feat.txt')
parser.add_argument('-training_filepath', type=str, default='meta/training_feat.txt')
parser.add_argument('-testing_filepath', type=str, default='meta/testing_feat.txt')

parser.add_argument('-input_dim', type=int, default=257, )
parser.add_argument('-num_classes', type=int, default=4)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=256)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=5)
parser.add_argument('-win_length', type=int, default=400)
parser.add_argument('-n_fft', type=int, default=512)
args = parser.parse_args()

### Data related
dataset_dev = SpeechDataGenerator(manifest=args.development_filepath, mode='dev', args=args)
dataloader_dev = DataLoader(dataset_dev, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate)

dataset_train = SpeechDataGenerator(manifest=args.training_filepath, mode='train', args=args)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate)

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath, mode='test', args=args)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate)



## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
speaker_feature_path = "/home/gdp/Desktop/Fatima_phd/mfcc_feature"

max_length =  50000
def extractFeature(dataloader, mode):
    count = 1
    feat_path = os.path.join(speaker_feature_path, mode)
    if os.path.exists(feat_path):
        shutil.rmtree(feat_path)
    os.mkdir(feat_path)

    for i_batch, sample_batched in enumerate(dataloader):
        print("Processing batch", count, mode)
        count = count + 1
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        paths = np.asarray([torch_tensor for torch_tensor in sample_batched[2]])
        features, labels = features.to(device), labels.to(device)
        numpy_features = features.detach().cpu().numpy()
        numpy_labels = labels.detach().cpu().numpy()
        writeFeatures(feat_path,numpy_labels, features, paths, mode)
        #################################################


if __name__ == '__main__':
    print(args)
    extractFeature(dataloader_dev, "dev")
    extractFeature(dataloader_train, "train")
    extractFeature(dataloader_test, "test")
