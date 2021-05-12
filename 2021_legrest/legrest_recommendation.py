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
3. Source data statistics
'''

'''
3.1 Basic Analysis
'''
# Subjects' age
if os.path.isfile("age_statistics.png")==False:
    source = data_config.loc[:, ['user_age']]
    plt.figure()
    plt.title('Subject Age')
    plt.xlabel('age (years)')
    plt.ylabel('count (subjects)')
    plt.hist(source)
    plt.grid()
    source.plot.kde()
    plt.savefig("age_statistics", bbox_inches='tight', pad_inches=0)


# Subjects' height
if os.path.isfile("height_statistics.png")==False:
    source = data_config.loc[:, ['user_height']]
    plt.figure()
    plt.title('Subject Height')
    plt.xlabel('height (cm)')
    plt.ylabel('count (subjects)')
    plt.hist(source)
    plt.grid()
    source.plot.kde()
    plt.savefig("height_statistics", bbox_inches='tight', pad_inches=0)


# Subjects' weight
if os.path.isfile("weight_statistics.png")==False:
    source = data_config.loc[:, ['user_weight']]
    plt.figure()
    plt.title('Subject Weight')
    plt.xlabel('weight (kg)')
    plt.ylabel('count (subjects)')
    plt.hist(source)
    plt.grid()
    source.plot.kde()
    plt.savefig("weight_statistics", bbox_inches='tight', pad_inches=0)


# Subjects' Bestfit(standard mode)
if os.path.isfile("standard_statistics.png")==False:
    source = data_config.loc[:, ['bestfit_angle_standard']]
    plt.figure()
    plt.title('Bestfit Angle @ Standard Mode')
    plt.xlabel('Angle (Raw ADC)')
    plt.hist(source)
    plt.grid()
    source.plot.kde()
    plt.savefig("standard_statistics", bbox_inches='tight', pad_inches=0)


# Subjects' Bestfit(relax mode)
if os.path.isfile("relax_statistics.png")==False:
    source = data_config.loc[:, ['bestfit_angle_relax']]
    plt.figure()
    plt.title('Bestfit Angle @ Relax Mode')
    plt.xlabel('Angle (Raw ADC)')
    plt.hist(source)
    plt.grid()
    source.plot.kde()
    plt.savefig("relax_statistics", bbox_inches='tight', pad_inches=0)


'''
3.2 Analysis : Correlation between age and bestfit
'''
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

source = data_config.loc[:, ['user_age', 'bestfit_angle_standard']]
age_20s = source.loc[(source['user_age']<30)&(source['user_age']>=20),'bestfit_angle_standard']
age_30s = source.loc[(source['user_age']<40)&(source['user_age']>=30),'bestfit_angle_standard']
age_40s = source.loc[(source['user_age']<50)&(source['user_age']>=40),'bestfit_angle_standard']

if os.path.isfile("age_bestfit_s_correlation.png")==False:
    plt.figure()
    plt.title("Age-Bestfit @ Standard Mode")
    plt.boxplot([age_20s, age_30s, age_40s], labels=['20s', '30s', '40s'], showmeans=True)
    plt.savefig("age_bestfit_s_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Ages-Bestfit@standard")
F_statistic, pVal = stats.f_oneway(age_20s, age_30s, age_40s) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


source = data_config.loc[:, ['user_age', 'bestfit_angle_relax']]
age_20s = source.loc[(source['user_age']<30)&(source['user_age']>=20),'bestfit_angle_relax']
age_30s = source.loc[(source['user_age']<40)&(source['user_age']>=30),'bestfit_angle_relax']
age_40s = source.loc[(source['user_age']<50)&(source['user_age']>=40),'bestfit_angle_relax']

if os.path.isfile("age_bestfit_r_correlation.png")==False:
    plt.figure()
    plt.title("Age-Bestfit @ Relax Mode")
    plt.boxplot([age_20s, age_30s, age_40s], labels=['20s', '30s', '40s'], showmeans=True)
    plt.savefig("age_bestfit_r_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Ages-Bestfit@relax")
F_statistic, pVal = stats.f_oneway(age_20s, age_30s, age_40s) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


'''
ANOVA analysis for bestfit@standard with age group
'''
#등분산성 검정 (bartlett test, fligner test, levene test)
# print(stats.bartlett(age_20s, age_30s, age_40s),stats.fligner(age_20s, age_30s, age_40s) ,stats.levene(age_20s, age_30s, age_40s), sep="\n")
# #정규성 검정 (Kilmogorov-Smirnov test)
# print(stats.ks_2samp(age_20s, age_30s), stats.ks_2samp(age_20s, age_40s), stats.ks_2samp(age_30s, age_40s),  sep="\n")


'''
ANOVA-height
'''
source = data_config.loc[:, ['user_height', 'bestfit_angle_standard']]
height_160 = source.loc[(source['user_height']<170)&(source['user_height']>=160),'bestfit_angle_standard']
height_170 = source.loc[(source['user_height']<180)&(source['user_height']>=170),'bestfit_angle_standard']
height_180 = source.loc[(source['user_height']<190)&(source['user_height']>=180),'bestfit_angle_standard']

if os.path.isfile("height_bestfit_s_correlation.png")==False:
    plt.figure()
    plt.title("Height-Bestfit @ Standard Mode")
    plt.boxplot([height_160, height_170, height_180], labels=['160s', '170s', '180s'], showmeans=True)
    plt.savefig("height_bestfit_s_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Height-Bestfit@standard")
F_statistic, pVal = stats.f_oneway(height_160, height_170, height_180) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


source = data_config.loc[:, ['user_height', 'bestfit_angle_relax']]
height_160 = source.loc[(source['user_height']<170)&(source['user_height']>=160),'bestfit_angle_relax']
height_170 = source.loc[(source['user_height']<180)&(source['user_height']>=170),'bestfit_angle_relax']
height_180 = source.loc[(source['user_height']<190)&(source['user_height']>=180),'bestfit_angle_relax']

if os.path.isfile("height_bestfit_r_correlation.png")==False:
    plt.figure()
    plt.title("Height-Bestfit @ Relax Mode")
    plt.boxplot([height_160, height_170, height_180], labels=['160s', '170s', '180s'], showmeans=True)
    plt.savefig("height_bestfit_r_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Height-Bestfit@relax")
F_statistic, pVal = stats.f_oneway(height_160, height_170, height_180) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


'''
ANOVA-weight
'''
source = data_config.loc[:, ['user_weight', 'bestfit_angle_standard']]
weight_50 = source.loc[(source['user_weight']<60)&(source['user_weight']>=50),'bestfit_angle_standard']
weight_60 = source.loc[(source['user_weight']<70)&(source['user_weight']>=60),'bestfit_angle_standard']
weight_70 = source.loc[(source['user_weight']<80)&(source['user_weight']>=70),'bestfit_angle_standard']
weight_80 = source.loc[(source['user_weight']<90)&(source['user_weight']>=80),'bestfit_angle_standard']

if os.path.isfile("weight_bestfit_s_correlation.png")==False:
    plt.figure()
    plt.title("Weight-Bestfit @ Standard Mode")
    plt.boxplot([weight_50, weight_60, weight_70, weight_80], labels=['50s', '60s', '70s', '80s'], showmeans=True)
    plt.savefig("weight_bestfit_s_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Weight-Bestfit@standard")
F_statistic, pVal = stats.f_oneway(weight_50, weight_60, weight_70, weight_80) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


source = data_config.loc[:, ['user_weight', 'bestfit_angle_standard']]
weight_50 = source.loc[(source['user_weight']<60)&(source['user_weight']>=50),'bestfit_angle_standard']
weight_60 = source.loc[(source['user_weight']<70)&(source['user_weight']>=60),'bestfit_angle_standard']
weight_70 = source.loc[(source['user_weight']<80)&(source['user_weight']>=70),'bestfit_angle_standard']
weight_80 = source.loc[(source['user_weight']<90)&(source['user_weight']>=80),'bestfit_angle_standard']

if os.path.isfile("weight_bestfit_r_correlation.png")==False:
    plt.figure()
    plt.title("Weight-Bestfit @ Relax Mode")
    plt.boxplot([weight_50, weight_60, weight_70, weight_80], labels=['50s', '60s', '70s', '80s'], showmeans=True)
    plt.savefig("weight_bestfit_r_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Weight-Bestfit@relax")
F_statistic, pVal = stats.f_oneway(weight_50, weight_60, weight_70, weight_80) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))