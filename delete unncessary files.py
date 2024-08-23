import glob
import shutil
import os

filePaths = glob.glob("C:\\uaspeech\\UASpeech\\audio\\M08\\fsr\\*.npy")
for path in filePaths:
    #print(path)
    os.remove(path)