import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_score, recall_score

LABELS = [0, 1]
sdtd = pd.read_csv("C:\\Users\\Talib\\Downloads\\talib\\talib\\X-vector_codes\\f1score\\sdtd_test_score.txt")
actual = sdtd.actual.tolist();
pred = sdtd.pred.tolist()
cm = confusion_matrix(actual, pred, labels=LABELS )

print("Accuracy  = ", accuracy_score(actual, pred))
print("F1 score = ", f1_score(actual,pred, average='macro'))
print("Precision = ", precision_score(actual,pred, average='macro'))
print("Recall = ", recall_score(actual,pred, average='macro'))