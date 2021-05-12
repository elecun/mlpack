'''
@brief  Legrest angle(Recline) recommendataion model
@author Byunghun Hwang <bh.hwang@iae.re.kr>
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import gc
from skimage import io, color
import tensorflow as tf
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPooling1D
from sklearn import svm
from sklearn.datasets._samples_generator import make_blobs
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import random
from sklearn import metrics
from sklearn.metrics import precision_score


'''
Presets & Hyper-parameters
'''
CONFIGURATION_FILE_PATH = "./data/train/data_config.csv"
DATASET_PATH = "./data/train/"
pd.set_option('display.width', 200) # for display width
DYNAMIC_SCALEUP = False
INTERPOLATION_METHOD = "catrom"
CASE_PATH = "./catrom_static"
SAVE_FEATURE_IMAGE = True
FEATURE_LENGTH = 30 # n-dimensional data feature only use
NUMBER_OF_SAMPLES = 299 # number of augmented data
mu, sigma = 0, 1 # normal distribution random parameter for data augmentation
FEATURE_MAX_LENGTH = 115 # Maximum feature length
NUMBER_OF_RANDOM_SELECTION = 5
MAX_TRAIN_ITERATION = -1 # infinity
SVM_KERNEL_METHOD = 'linear'
NUMBER_OF_TESTING = 10
IMAGE_HEIGHT = 369



'''
1. Load configuration file
'''
data_config = pd.read_csv(CONFIGURATION_FILE_PATH, header=0, index_col=0)

fsr_dataframe = {}
seat_dataframe = {}

for idx in data_config.index:
    fsr_filepath = DATASET_PATH+data_config.loc[idx, "fsr_matrix_1d_datafile"] # set FSR matrix data filepath
    seat_filepath = DATASET_PATH+data_config.loc[idx, "seat_datafile"] # set Seat data filepath
    print(idx, ") read data files : ", fsr_filepath, ",", seat_filepath)

    fsr_dataframe[idx] = pd.read_csv(fsr_filepath, header=0, index_col=False).iloc[:,0:162] # read FSR matrix data file
    seat_dataframe[idx] = pd.read_csv(seat_filepath, header=0, index_col=False) # read Seat data file

    # clear unnecessary columns
    del seat_dataframe[idx]['Measurement time'] # remove unnecessary column
    del fsr_dataframe[idx]['Measurement Time (sec)'] # remove unnecessary column


'''
2. Source data segmentation
'''
fsr_dataframe_standard_segment = {}
fsr_dataframe_relax_segment = {}
seat_loadcell_dataframe_standard_segment = {}
seat_loadcell_dataframe_relax_segment = {}

for idx in data_config.index:
    mtime = data_config.loc[idx, ['standard_s_mtime', "standard_e_mtime", "relax_s_mtime", "relax_e_mtime"]]

    # seat loadcell segmentation
    seat_loadcell_dataframe_standard_segment[idx] = seat_dataframe[idx][(seat_dataframe[idx]['mtime']>=mtime.standard_s_mtime) & (seat_dataframe[idx]['mtime']<=mtime.standard_e_mtime)]
    seat_loadcell_dataframe_relax_segment[idx] = seat_dataframe[idx][(seat_dataframe[idx]['mtime']>=mtime.relax_s_mtime) & (seat_dataframe[idx]['mtime']<=mtime.relax_e_mtime)]

    # fsr matrix segmentation
    fsr_dataframe_standard_segment[idx] = fsr_dataframe[idx][(fsr_dataframe[idx]['mtime']>=mtime.standard_s_mtime) & (fsr_dataframe[idx]['mtime']<=mtime.standard_e_mtime)]
    fsr_dataframe_relax_segment[idx] = fsr_dataframe[idx][(fsr_dataframe[idx]['mtime']>=mtime.relax_s_mtime) & (fsr_dataframe[idx]['mtime']<=mtime.relax_e_mtime)]

    print("FSR Segments@Standard size : ", len(fsr_dataframe_standard_segment[idx]), ", FSR Segments@Relax size : ", len(fsr_dataframe_relax_segment[idx]))
    print("Seat Segments@Standard size : ", len(seat_loadcell_dataframe_standard_segment[idx]), ", Seat Segments@Relax size : ", len(seat_loadcell_dataframe_relax_segment[idx]))


'''
Pearson Correlation
'''
import scipy.stats as stats

# height
source = data_config.loc[:, ['user_height', 'bestfit_angle_standard']]
corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['user_height'], source['bestfit_angle_standard']))

source = data_config.loc[:, ['user_height', 'bestfit_angle_relax']]
corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['user_height'], source['bestfit_angle_relax']))

# weight
source = data_config.loc[:, ['user_weight', 'bestfit_angle_standard']]
corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['user_weight'], source['bestfit_angle_standard']))

source = data_config.loc[:, ['user_weight', 'bestfit_angle_relax']]
corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['user_weight'], source['bestfit_angle_relax']))


# age
source = data_config.loc[:, ['user_age', 'bestfit_angle_standard']]
corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['user_age'], source['bestfit_angle_standard']))

source = data_config.loc[:, ['user_age', 'bestfit_angle_relax']]
corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['user_age'], source['bestfit_angle_relax']))


# bmi
source = data_config.loc[:, ['user_weight','user_height', 'bestfit_angle_standard', 'bestfit_angle_relax']]
bmi = source['user_weight']/(source['user_height']/100*source['user_height']/100)
source["bmi"] = bmi

corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['bmi'], source['bestfit_angle_standard']))

corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['bmi'], source['bestfit_angle_relax']))


# bmr
source = data_config.loc[:, ['user_weight','user_height', 'bestfit_angle_standard', 'user_age', 'bestfit_angle_relax']]
bmr = 66.47+(13.75*source['user_weight'])+(5*source['user_height'])-(6.76*source['user_age'])
source["bmr"] = bmr

corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['bmr'], source['bestfit_angle_standard']))

corr = source.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(source['bmr'], source['bestfit_angle_relax']))