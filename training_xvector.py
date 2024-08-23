import torch
import numpy as np
from torch.utils.data import DataLoader
from XvecSpeechGenerator import XvecSpeechGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate
from utils.utils import writeXvectors
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import shutil
import matplotlib.pyplot as plt


########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-dev_filepath',type=str,default='meta/dev_feat.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing_feat.txt')
parser.add_argument('-training_filepath',type=str, default='meta/training_feat.txt')

parser.add_argument('-input_dim', type=int, default=257,)
parser.add_argument('-num_classes', type=int, default=15)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=10)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=20)
parser.add_argument('-win_length', type=int, default=400)
parser.add_argument('-n_fft', type=int, default=512)
args = parser.parse_args()

### Data related
dataset_dev = XvecSpeechGenerator(manifest=args.dev_filepath,mode='dev', args=args)
dataloader_dev = DataLoader(dataset_dev, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate)

dataset_train = XvecSpeechGenerator(manifest=args.training_filepath, mode='train',args=args)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate)


dataset_test = XvecSpeechGenerator(manifest=args.testing_filepath,mode='test',args=args)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate)

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(args.input_dim, args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()


speaker_xvec_path = "C:\\Users\\Talib\\Desktop\\xvec_data\\xvecs\\"
def train(dataloader_train,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    count = 1
    x_vec = ""
    path = "xvector"
    df = pd.DataFrame(columns=['x_vec_path', 'label'])

    if os.path.exists(path) == False:
        os.mkdir(path)

    xvec_feat_path = os.path.join(speaker_xvec_path, "dev")
    if os.path.exists(xvec_feat_path):
        shutil.rmtree(xvec_feat_path)
    os.mkdir(xvec_feat_path)

    for i_batch, sample_batched in enumerate(dataloader_train):
        print("Processing batch" , "Dev ", count)
        count = count + 1
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        paths = np.asarray([torch_tensor for torch_tensor in sample_batched[2]])
        print("Processing files ", paths)
        features, labels = features.to(device),labels.to(device)
        numpy_features = features.detach().cpu().numpy()
        numpy_labels = labels.detach().cpu().numpy()
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits,x_vec = model(features)
        #################################################
        ############ Writing X Vectors ##################
        if epoch == args.num_epochs - 1:
            print("Started writing xvecs at epoch count", epoch)
            df = writeXvectors(xvec_feat_path , df, numpy_labels, x_vec, paths)
        #################################################
        #### CE loss
        loss = loss_fun(pred_logits,labels.long())
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        count = count + 1
        #train_acc_list.append(accuracy)
        #if i_batch%10==0:
        #    print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))

        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)

    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    #print("Training xvector ", x_vec.detach().numpy())
    path = os.path.join(path, "dev_x_vec.npy")
    np.save(path, x_vec.cpu().detach().numpy())
    df.to_csv(os.path.join(xvec_feat_path, "dev_xvect_feat.csv"), encoding='utf-8')
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
    rand = np.random.rand()
    model_save_path = os.path.join('save_model')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    path = os.path.join(model_save_path, 'train_best_check_point_' + str(epoch) + '_' + str(mean_loss) + '_' + str(rand))
    state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(model.state_dict(), path)
    return (mean_loss, mean_acc)



def validation(dataloader_val, mode):
    #model1 = torch.load("C:\\Users\\Talib\\Downloads\\talib\\talib\\X-vector_codes\\save_model\\best_check_point_1_1.395266056060791_0.761649240146505")
    model = X_vector(args.input_dim, args.num_classes).to(device)
    model.state_dict(torch.load("save_model/train_best_check_point_19_0.5142491459846497_0.5368522291856418"))
    print("Processing mode {}".format( mode))
    #model.eval()

    path = "xvector"
    count = 0
    xvec_feat_path = ""
    df = pd.DataFrame(columns=['x_vec_path', 'label'])
    if mode == "dev":
        xvec_feat_path = os.path.join(speaker_xvec_path, "dev")
    elif mode == "train":
        xvec_feat_path = os.path.join(speaker_xvec_path, "train")
    else:
        xvec_feat_path = os.path.join(speaker_xvec_path, "test")
    if os.path.exists(xvec_feat_path):
        shutil.rmtree(xvec_feat_path)
    os.mkdir(xvec_feat_path)
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            paths = np.asarray([torch_tensor for torch_tensor in sample_batched[2]])
            print("Processing files {} and batch count {}".format(paths, count))
            numpy_features = features.detach().cpu().numpy()
            numpy_labels = labels.detach().cpu().numpy()
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #################################################
            ############ Writing X Vectors ##################
            df = writeXvectors(xvec_feat_path, df, numpy_labels, x_vec, paths)
            #################################################
            #### CE loss
            loss = loss_fun(pred_logits,labels.long())
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            count = count + 1
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)


        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        #print("Training xvector ", x_vec.detach().numpy())
        if mode == "dev":
            path = os.path.join(path, "dev_x_vec.npy")
        elif mode == "test":
            path = os.path.join(path, "test_x_vec.npy")
        else:
            path = os.path.join(path, "train_x_vec.npy")
        np.save(path, x_vec.cpu().detach().numpy())
        df.to_csv(os.path.join(xvec_feat_path, "train_xvect_feat.csv"), encoding='utf-8')
        print('Total vlidation loss {} and Validation accuracy {} after'.format(mean_loss,mean_acc))



if __name__ == '__main__':
    print(args)
    trainloss = []
    train_acc = []
    '''
    for epoch in range(args.num_epochs):
            loss, acc = train(dataloader_dev,epoch)
            trainloss.append(loss)
            train_acc.append(acc)
        #validation(dataloader_train, epoch, "train")
        #validation(dataloader_test, epoch, "test")
    plt.plot(trainloss)
    plt.show()
    '''
    validation(dataloader_dev, "dev")
    validation(dataloader_train, "train")
    validation(dataloader_test, "test")
