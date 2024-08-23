#!/bin/bash

# Run the main programs. For help, run the following command:
#       python main.py -h
#python main.py --data_info specify --data_path /home/mrinmoy/Projects/SR_MEiTY/IITG-MV_database/IITG_MV_Phase-I_Speaker_Recognition_Database/100_speaker_Database_Office_Environment/ --output_path ../


#python data_preparation.py
#python feature_computation.py
#python train_test_model.py --model_type GMM-UBM
#python train_test_split.py
#python xvector2.py
#python speakerwise_xvector.py
###############################################################################################################################################
# New X-vector system
###############################################################################################################################################

python FeatureExtractor.py -development_filepath /home/gdp/Desktop/Fatima_phd/file_list/dev.txt -training_filepath /home/gdp/Desktop/Fatima_phd/file_list/train_ua.txt -testing_filepath /home/gdp/Desktop/Fatima_phd/file_list/test_sdtd.txt -input_dim 40 -num_classes 4 -win_length 400 -n_fft 400

###python training_xvector.py -dev_filepath /home/gdp/Desktop/SVS_PY/feature_mfcc/dev/dev.text -testing_filepath /home/gdp/Desktop/SVS_PY/feature_mfcc/test/test.text -training_filepath /home/gdp/Desktop/SVS_PY/feature_mfcc/train/train.text -input_dim 40 -num_classes 60 -win_length 78 -n_fft 78

#python plda_train.py

#python speaker_wise_average.py


#python plda_testing.py
#python test.py



