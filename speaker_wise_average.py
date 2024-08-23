import os
import glob
import numpy as np
import ntpath
import shutil
xvec_base_path = "C:\\Users\\Talib\\Desktop\\xvec_data\\xvecs\\*"
average_speaker_folder = "C:\\Users\\Talib\\Desktop\\xvec_data\\xvecs\\average_vecs\\"
envs = glob.glob(xvec_base_path)

if os.path.exists(average_speaker_folder):
    shutil.rmtree(average_speaker_folder)
os.mkdir(average_speaker_folder)

for env in envs:
    subdirs = glob.glob(env + "/*")
    for subdir in subdirs:
        if os.path.isdir(subdir):
            list = []
            count = 0
            speakerId = ""
            filePaths = glob.glob(subdir + "/*.npy")
            for file in filePaths:
                data = np.load(file)
                data = np.reshape(data, (1, 512))
                data = np.mean(data, axis=0, keepdims=True)
                list.append(data)
                if count == 0:
                    base_file_name = ntpath.basename(file)
                    speakerId = base_file_name.split("_")[0]
                    count = count + 1

            arr = np.array(list)
            mean = np.mean(arr, axis=0)
            path = os.path.join(average_speaker_folder, speakerId + ".npy")
        
            np.save(path, np.ravel(mean))
