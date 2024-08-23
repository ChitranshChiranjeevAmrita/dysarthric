import pickle
import numpy as np
import pandas as pd
import os

filename = 'C:\\Users\\Talib\\Downloads\\talib\\talib\\X-vector_codes\\pldamodels\\train_plda_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#
train_feat_file = pd.read_csv("/dysarthric/pldatestfiles/mfcc/sdtd.csv")
x_vec1 = []
x_vec2 = []
for i in range(len(train_feat_file.p1.tolist())):
    x_vec1.append(np.load(train_feat_file.p1[i]))
    x_vec2.append(np.load(train_feat_file.p2[i]))

U_model1 = loaded_model.model.transform(np.array(x_vec1), from_space='D', to_space='U_model')
U_model2 = loaded_model.model.transform(np.array(x_vec2), from_space='D', to_space='U_model')


#train_feat_file = pd.read_csv("C:\\Users\\Talib\\Downloads\\talib\\talib\\X-vector_codes\\yhipe.csv")
pred_score = []
pred_cat = []
count = 0
for i in range(len(train_feat_file.p1.tolist())):
    score = loaded_model.model.calc_same_diff_log_likelihood_ratio(U_model1[i][None, ], U_model2[i][None, ])
    pcat = ""
    if score >= 0:
        pcat = 1
    else:
        pcat = 0
    pred_cat.append(pcat)
    pred_score.append(score)
    if pcat == train_feat_file.iloc[i]["GT"]:
        count = count + 1

train_feat_file["pred_score"] = pred_score
train_feat_file["pred_label"] = pred_cat

train_feat_file.to_csv("score_sdtd.csv")
print(train_feat_file)
print("Accuracy = ", count / len(train_feat_file.p1.tolist()))