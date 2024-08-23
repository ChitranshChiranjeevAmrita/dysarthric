import pywt  # for wavelet transform and thresholding
import numpy as np

# Decompose the speech signal using VMD
modes = vmdpy.decompose(signal)  # Assuming 'vmdpy' is the module for VMD

threshold = 0.5  # Threshold value

thresholded_modes = []  # List to store thresholded modes

for mode in range(1, len(modes)):
    # Perform 3-level DWT decomposition
    coeffs = pywt.wavedec(mode, 'db4', level=3)

    # Apply thresholding to each level of decomposition
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    # Reconstruct the mode using inverse DWT
    thresholded_mode = pywt.waverec(coeffs, 'db4')

    thresholded_modes.append(thresholded_mode)

reconstructed_signal = np.sum(thresholded_modes, axis=0)
