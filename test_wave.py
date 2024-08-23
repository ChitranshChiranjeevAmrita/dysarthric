import numpy as np


arr1 = np.load("C:\\Users\\Talib\\Desktop\\xvec_data\\mfccs\\test\\1\\test_private_0093.npy")
arr2 = np.load("C:\\Users\\Talib\\Desktop\\xvec_data\\mfccs\\test\\1\\test_private_0091.npy")
#arr = np.mean(arr, axis = 0)
arr=arr1-arr2
print(arr)
print(arr.shape)