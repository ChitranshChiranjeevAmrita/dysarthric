import os
import glob
import ntpath

# xvec_base_path = "C:\\Users\\Talib\\Downloads\\headshots"
# envs = glob.glob(xvec_base_path)
# count = 1
# for env in envs:
#     subdirs = glob.glob(env + "/*")
#     for subdir in subdirs:
#         if os.path.isdir(subdir):
#             print("Team Name is : ", subdir)
#             filePaths = glob.glob(subdir + "/*.PNG")
#             optaIds = []
#             for file in filePaths:
#                     base_file_name = ntpath.basename(file)
#                     optaIds.append(base_file_name.split(".")[0])
#                     count = count + 1
#             print(optaIds)
# print("Total Count = ", count)

xvec_base_path = "C:\\Users\\Talib\\Downloads\\WSL\\WSL"
envs = glob.glob(xvec_base_path)
count = 1
optaIds = []
for env in envs:
    subdirs = glob.glob(env + "/*")
    for subdir in subdirs:
        base_file_name = ntpath.basename(subdir)
        optaIds.append(int(base_file_name.split(".")[0]))
print("Total Count = ", optaIds)