# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import librosa as lbr
import IPython.display as ipd
import librosa


#path_hi="C:\\uaspeech\\UASpeech\\audio\\F05\\fsr\\F05_B1_UW54_M2_FSR.wav"
path_hi = "C:\\uaspeech\\UASpeech\\audio\\F03\\fsr\\F03_B1_UW54_M2_FSR.wav"

speech_signal , sr = lbr.load(path_hi, sr=None)
input_length = sr
if len(speech_signal) > input_length:
    max_offset = len(speech_signal) - input_length
    offset = np.random.randint(max_offset)
    speech_signal = speech_signal[offset:(input_length+offset)]
else:
    if input_length > len(speech_signal):
        max_offset = input_length - len(speech_signal)
        offset = np.random.randint(max_offset)
    else:
            offset = 0
    speech_signal = np.pad(speech_signal, (offset, input_length - len(speech_signal) - offset), "constant")

x = speech_signal
x /= np.abs(x).max()

#frame_length = 320
frame_length = 0.025  # Frame length in seconds
hop_length = 0.010  # Hop length in seconds

frame_length_samples = int(frame_length * sr)
print("frame_length_samples = ", frame_length_samples)
hop_length_samples = int(hop_length * sr)
print("hop_length_samples = ", hop_length_samples)


fig = plt.figure(figsize=(60, 40))
plt.suptitle("LP error power plot", fontsize=18)
a = 20
b = 2
c = 1
for L in range(1, 11):
    len0 = np.max(np.size(x))
    e = np.zeros(np.size(x))  # prediction error variable initialization
    #blocks = np.int(np.floor(len0 / frame_length))  # total number of blocks
    # Calculate the number of frames
    blocks = 1 + (len(x) - frame_length_samples) // hop_length_samples
    print("Blocks ", blocks)
    state = np.zeros(L)  # Memory state of prediction filter
    # Building our Matrix A from blocks of length 640 samples and process:
    h = np.zeros((blocks, L))  # initialize pred. coeff memory
    power = np.zeros(blocks)
    for m in range(0, blocks):
        A = np.zeros((frame_length_samples - L, L))  # trick: up to 320 to avoid zeros in the matrix
        for n in range(0, frame_length_samples - L):
            #A[n, :] = np.flipud(x[m * frame_length + n + np.arange(L)])
            A[n, :] = np.flipud(x[m * hop_length_samples + n + np.arange(L)])

        # Construct our desired target signal d, one sample into the future:
        d = x[m * hop_length_samples + np.arange(L, frame_length_samples)];
        # Compute the prediction filter:
        h[m, :] = np.dot(np.dot(np.linalg.pinv(np.dot(A.transpose(), A)), A.transpose()), d)
        hperr = np.hstack([1, -h[m, :]])
        e[m * hop_length_samples + np.arange(0, frame_length_samples)], state = sp.lfilter(hperr, [1], x[
            m * hop_length_samples + np.arange(0, frame_length_samples)], zi=state)
        # e/=np.abs(e).max()
        p = e[m * hop_length_samples + np.arange(0, frame_length_samples)]
        power[m] = np.dot(p.transpose(), p) / np.max(np.size(p))
        # power = librosa.amplitude_to_db(power, ref=np.max)
        # print("Power = ", np.dot(p.transpose(),p)/np.max(np.size(p)))
    plt.subplot(a, b, c)
    # plt.title('Signal: {}, order: {}'.format(path_hi, c))
    # plt.xlabel(i)
    plt.plot(x, color='b')
    plt.plot(e, color='r')
    c = c + 1

    plt.subplot(a, b, c)
    # plt.title('Signal: {}, order: {}'.format(path_hi, c))
    # plt.xlabel(i)
    plt.plot(power, color='g')
    # plt.plot(e, color='r')
    c = c + 1
# plt.tight_layout()
plt.show()