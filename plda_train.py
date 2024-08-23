import pickle

import plda
import pandas as pd
import numpy as np
import os
import shutil

modelPath = "pldamodels"

if os.path.exists(modelPath):
    shutil.rmtree(modelPath)
os.mkdir(modelPath)

xvecs_base_path = "C:\\Users\\Talib\\Desktop\\xvec_data\\xvecs"
train_feat_file = pd.read_csv("C:\\xvecs\\train\\train_xvect_feat.csv")

train_xvecs = []
train_labels = []
for index, (fpath, label) in enumerate(zip(train_feat_file['x_vec_path'].tolist(), train_feat_file['label'].tolist())):
    x_vec = np.load(os.path.abspath(fpath))
    #x_vec = np.reshape(x_vec, (1, 512))
    train_xvecs.append(x_vec)
    train_labels.append(label)

#Don't use this, this is degrading the performance.
#train_xvecs = StandardScaler().fit_transform(train_xvecs)
myModel = plda.Classifier()

# Pay attention here, how to choose n_principal_components
myModel.fit_model(np.array(train_xvecs), np.array(train_labels))

filename = 'C:\\Users\\Talib\\Downloads\\talib\\talib\\X-vector_codes\\pldamodels\\train_plda_model.sav'


pickle.dump(myModel, open(filename, 'wb'))
#U_model = model.model.transform(np.array(train_xvecs), from_space='D', to_space='U_model')




