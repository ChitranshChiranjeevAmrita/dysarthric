import numpy as np
from matplotlib import pyplot as plt


avg1 = np.load("C:\\xvec_features\\spectrogram\\train\\2\\F04_B1_UW8_M5_FSR.npy", allow_pickle=True)
avg2 = np.load("C:\\xvecs\\avergae\\vl.npy", allow_pickle=True)
vl = np.load("C:\\xvecs\\test\\0\\F03_B2_UW36_M4_FSR.npy", allow_pickle=True)
#hi = np.load("C:\\xvecs\\test\\3\\F05_B2_UW73_M4_FSR.npy", allow_pickle=True)
#arr = np.mean(arr, axis = 0)

plt.plot(avg1, 'r')
plt.plot(avg2, 'g')

#plt.plot(hi.flatten(), 'r')
#plt.plot(vl.flatten(), 'b')
#plt.plot(hi.flatten(), 'g')

plt.show()
print("Hello")
