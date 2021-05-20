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

# relationship between height and bestfit@standard & relax
if os.path.isfile("height_bestfit_relationship.png")==False:
    source = data_config.loc[:, ['user_height', 'bestfit_angle_standard', 'bestfit_angle_relax']]
    plt.figure()
    plt.title('Relationship between Height and Bestfit')
    plt.xlabel('Height(cm)')
    plt.ylabel('Bestfit')
    plt.scatter(source["user_height"], source["bestfit_angle_standard"])
    plt.scatter(source["user_height"], source["bestfit_angle_relax"]+200)
    plt.grid()
    plt.legend(labels=["standard", "relax"])
    plt.savefig("height_bestfit_relationship", bbox_inches='tight', pad_inches=0)


if os.path.isfile("age_bestfit_relationship.png")==False:
    source = data_config.loc[:, ['user_age', 'bestfit_angle_standard', 'bestfit_angle_relax']]
    plt.figure()
    plt.title('Relationship between Ages and Bestfit')
    plt.xlabel('Age(years)')
    plt.ylabel('Bestfit')
    plt.scatter(source["user_age"], source["bestfit_angle_standard"])
    plt.scatter(source["user_age"], source["bestfit_angle_relax"]+200)
    plt.grid()
    plt.legend(labels=["standard", "relax"])
    plt.savefig("age_bestfit_relationship", bbox_inches='tight', pad_inches=0)

if os.path.isfile("weight_bestfit_relationship.png")==False:
    source = data_config.loc[:, ['user_weight', 'bestfit_angle_standard', 'bestfit_angle_relax']]
    plt.figure()
    plt.title('Relationship between Weight and Bestfit')
    plt.xlabel('Weight(kg)')
    plt.ylabel('Bestfit')
    plt.scatter(source["user_weight"], source["bestfit_angle_standard"])
    plt.scatter(source["user_weight"], source["bestfit_angle_relax"]+200)
    plt.grid()
    plt.legend(labels=["standard", "relax"])
    plt.savefig("weight_bestfit_relationship", bbox_inches='tight', pad_inches=0)

if os.path.isfile("bmi_bestfit_relationship.png")==False:
    source = data_config.loc[:, ['user_height','user_weight', 'bestfit_angle_standard', 'bestfit_angle_relax']]
    bmi = source['user_weight']/(source['user_height']/100*source['user_height']/100)
    plt.figure()
    plt.title('Relationship between BMI and Bestfit')
    plt.xlabel('BMI')
    plt.ylabel('Bestfit')
    plt.scatter(bmi, source["bestfit_angle_standard"])
    plt.scatter(bmi, source["bestfit_angle_relax"]+200)
    plt.grid()
    plt.legend(labels=["standard", "relax"])
    plt.savefig("bmi_bestfit_relationship", bbox_inches='tight', pad_inches=0)


if os.path.isfile("bmr_bestfit_relationship.png")==False:
    source = data_config.loc[:, ['user_age', 'user_height','user_weight', 'bestfit_angle_standard', 'bestfit_angle_relax']]
    bmr = 66.47+(13.75*source['user_weight'])+(5*source['user_height'])-(6.76*source['user_age'])
    plt.figure()
    plt.title('Relationship between BMR and Bestfit')
    plt.xlabel('BMR')
    plt.ylabel('Bestfit')
    plt.scatter(bmr, source["bestfit_angle_standard"])
    plt.scatter(bmr, source["bestfit_angle_relax"]+200)
    plt.grid()
    plt.legend(labels=["standard", "relax"])
    plt.savefig("bmr_bestfit_relationship", bbox_inches='tight', pad_inches=0)


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


'''
ANOVA-gender
'''
source = data_config.loc[:, ['user_gender', 'bestfit_angle_standard']]
female = source.loc[(source['user_gender']==0),'bestfit_angle_standard']
male = source.loc[(source['user_gender']==1),'bestfit_angle_standard']

if os.path.isfile("gender_bestfit_s_correlation.png")==False:
    plt.figure()
    plt.title("Gender-Bestfit @ Standard Mode")
    plt.boxplot([male, female], labels=['male', 'female'], showmeans=True)
    plt.savefig("gender_bestfit_s_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Gender-Bestfit@standard")
F_statistic, pVal = stats.f_oneway(male, female) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


source = data_config.loc[:, ['user_gender', 'bestfit_angle_relax']]
female = source.loc[(source['user_gender']==0),'bestfit_angle_relax']
male = source.loc[(source['user_gender']==1),'bestfit_angle_relax']

if os.path.isfile("gender_bestfit_r_correlation.png")==False:
    plt.figure()
    plt.title("Gender-Bestfit @ Relax Mode")
    plt.boxplot([male, female], labels=['male', 'female'], showmeans=True)
    plt.savefig("gender_bestfit_r_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : Gender-Bestfit@relax")
F_statistic, pVal = stats.f_oneway(male, female) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))



'''
ANOVA-BMI
'''
source = data_config.loc[:, ['user_height','user_weight', 'bestfit_angle_standard']]
bmi = source['user_weight']/(source['user_height']/100*source['user_height']/100)
bmi_source = pd.DataFrame(source)
bmi_source['bmi'] = bmi

group1 = bmi_source.loc[bmi_source['bmi']<18.5,'bestfit_angle_standard']
group2 = bmi_source.loc[(bmi_source['bmi']<22.9)&(bmi_source['bmi']>=18.5),'bestfit_angle_standard']
group3 = bmi_source.loc[(bmi_source['bmi']<24.9)&(bmi_source['bmi']>=22.9),'bestfit_angle_standard']
group4 = bmi_source.loc[(bmi_source['bmi']<29.9)&(bmi_source['bmi']>=24.9),'bestfit_angle_standard']
group5 = bmi_source.loc[(bmi_source['bmi']<34.9)&(bmi_source['bmi']>=29.9),'bestfit_angle_standard']


if os.path.isfile("bmi_bestfit_s_correlation.png")==False:
    plt.figure()
    plt.title("BMI-Bestfit @ Standard Mode")
    plt.boxplot([group1, group2, group3, group4, group5], labels=['Group 1','Group 2', 'Group 3', 'Group 4', 'Group 5'], showmeans=True)
    plt.savefig("bmi_bestfit_s_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : BMI-Bestfit@standard")
F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4, group5) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


source = data_config.loc[:, ['user_height','user_weight', 'bestfit_angle_relax']]
bmi = source['user_weight']/(source['user_height']/100*source['user_height']/100)
bmi_source = pd.DataFrame(source)
bmi_source['bmi'] = bmi

group1 = bmi_source.loc[bmi_source['bmi']<18.5,'bestfit_angle_relax']
group2 = bmi_source.loc[(bmi_source['bmi']<22.9)&(bmi_source['bmi']>=18.5),'bestfit_angle_relax']
group3 = bmi_source.loc[(bmi_source['bmi']<24.9)&(bmi_source['bmi']>=22.9),'bestfit_angle_relax']
group4 = bmi_source.loc[(bmi_source['bmi']<29.9)&(bmi_source['bmi']>=24.9),'bestfit_angle_relax']
group5 = bmi_source.loc[(bmi_source['bmi']<34.9)&(bmi_source['bmi']>=29.9),'bestfit_angle_relax']


if os.path.isfile("bmi_bestfit_r_correlation.png")==False:
    plt.figure()
    plt.title("BMI-Bestfit @ Relax Mode")
    plt.boxplot([group1, group2, group3, group4, group5], labels=['Group 1','Group 2', 'Group 3', 'Group 4', 'Group 5'], showmeans=True)
    plt.savefig("bmi_bestfit_r_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : BMI-Bestfit@relax")
F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4, group5) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


'''
ANOVA-BMR
'''
source = data_config.loc[:, ['user_height','user_weight', 'user_age', 'user_gender', 'bestfit_angle_standard']]
bmr = 66.47+(13.75*source['user_weight'])+(5*source['user_height'])-(6.76*source['user_age'])
bmr_source = pd.DataFrame(source)
bmr_source['bmr'] = bmr

#group1 = bmr_source.loc[bmr_source['bmr']<1200,'bestfit_angle_standard']
group1 = bmr_source.loc[(bmr_source['bmr']<1300)&(bmr_source['bmr']>=1200),'bestfit_angle_standard']
group2 = bmr_source.loc[(bmr_source['bmr']<1400)&(bmr_source['bmr']>=1300),'bestfit_angle_standard']
group3 = bmr_source.loc[(bmr_source['bmr']<1500)&(bmr_source['bmr']>=1400),'bestfit_angle_standard']
group4 = bmr_source.loc[(bmr_source['bmr']<1600)&(bmr_source['bmr']>=1500),'bestfit_angle_standard']
group5 = bmr_source.loc[(bmr_source['bmr']<1700)&(bmr_source['bmr']>=1600),'bestfit_angle_standard']


if os.path.isfile("bmr_bestfit_s_correlation.png")==False:
    plt.figure()
    plt.title("BMR-Bestfit @ Standard Mode")
    plt.boxplot([group1, group2, group3, group4, group5], labels=['Group 1','Group 2', 'Group 3', 'Group 4', 'Group 5'], showmeans=True)
    plt.savefig("bmr_bestfit_s_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : BMR-Bestfit@standard")
F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4, group5) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


source = data_config.loc[:, ['user_height','user_weight', 'user_age', 'user_gender', 'bestfit_angle_relax']]
bmr = 66.47+(13.75*source['user_weight'])+(5*source['user_height'])-(6.76*source['user_age'])
bmr_source = pd.DataFrame(source)
bmr_source['bmr'] = bmr

# group1 = bmr_source.loc[bmr_source['bmr']<1200,'bestfit_angle_relax']
group1 = bmr_source.loc[(bmr_source['bmr']<1300)&(bmr_source['bmr']>=1200),'bestfit_angle_relax']
group2 = bmr_source.loc[(bmr_source['bmr']<1400)&(bmr_source['bmr']>=1300),'bestfit_angle_relax']
group3 = bmr_source.loc[(bmr_source['bmr']<1500)&(bmr_source['bmr']>=1400),'bestfit_angle_relax']
group4 = bmr_source.loc[(bmr_source['bmr']<1600)&(bmr_source['bmr']>=1500),'bestfit_angle_relax']
group5 = bmr_source.loc[(bmr_source['bmr']<1700)&(bmr_source['bmr']>=1600),'bestfit_angle_relax']


if os.path.isfile("bmr_bestfit_r_correlation.png")==False:
    plt.figure()
    plt.title("BMR-Bestfit @ Relax Mode")
    plt.boxplot([group1, group2, group3, group4, group5], labels=['Group 1','Group 2', 'Group 3', 'Group 4', 'Group 5'], showmeans=True)
    plt.savefig("bmr_bestfit_r_correlation", bbox_inches='tight', pad_inches=0)

print("ANOVA : BMR-Bestfit@relax")
F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4, group5) #Atman910
print('Altman 910 oneway ANOVA : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))


'''
Regression Analysis : OLS
'''
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

source = data_config.loc[:, ['user_height', 'user_weight', 'user_age', 'bestfit_angle_standard']]
source.boxplot(column = 'bestfit_angle_standard', by='user_weight' , grid=False)

formula = 'bestfit_angle_standard ~ user_height + user_weight + C(user_age)'
lm = ols(formula, data=source).fit()
print("---OLS bestfit@standard---")
print(anova_lm(lm))
print(lm.summary())


source = data_config.loc[:, ['user_height', 'user_weight', 'user_age', 'bestfit_angle_relax']]
source.boxplot(column = 'bestfit_angle_relax', by='user_weight' , grid=False)

formula = 'bestfit_angle_relax ~ user_height + user_weight + C(user_age)'
lm = ols(formula, data=source).fit()
print("---OLS bestfit@relax---")
print(anova_lm(lm))
print(lm.summary())

# results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
# print(results.summary())


'''
SVM Regression
'''

# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# X = data_config.loc[:, ['user_height', 'user_weight', 'user_age']]
# bmr = 66.47+(13.75*X['user_weight'])+(5*X['user_height'])-(6.76*X['user_age'])
# bmi = X['user_weight']/(X['user_height']/100*X['user_height']/100)
# X['bmr'] = bmr
# X['bmi'] = bmi
# y = data_config.loc[:, ['bestfit_angle_standard']]


# X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.33)
# regressor = make_pipeline(StandardScaler(), SVR(kernel="poly", degree=5, C=1.0, epsilon=0.3))

# regressor.fit(X_train,y_train)
# r = regressor.predict(X_test)
# print(r)
# print(X_test, y_test)
# print(regressor.score(X_test,y_test))



#################

# print(X.shape)
# print(X.to_numpy())
# print(y.shape)
# y = np.ravel(y)
# print(y)
# print(len(X))


# Xcon_train = np.array([], dtype=np.int64).reshape(0, 3)
# Xcon_test = np.array([], dtype=np.int64).reshape(0, 3)
# ycon_train = np.array([], dtype=np.int64)
# ycon_test = np.array([], dtype=np.int64)




# SVR Model
# lw = 2
# svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# svr_lin = SVR(kernel='linear', C=100, gamma='auto')
# svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

# svrs = [svr_rbf, svr_lin, svr_poly]
# kernel_label = ['RBF', 'Linear', 'Polynomial']
# model_color = ['m', 'c', 'g']

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
# for ix, svr in enumerate(svrs):
#     axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw, label='{} model'.format(kernel_label[ix]))
#     axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none", edgecolor=model_color[ix], s=50,
#                      label='{} support vectors'.format(kernel_label[ix]))
#     axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
#                      y[np.setdiff1d(np.arange(len(X)), svr.support_)],
#                      facecolor="none", edgecolor="k", s=50, label='other training data')
#     axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)

# fig.text(0.5, 0.04, 'data', ha='center', va='center')
# fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
# fig.suptitle("Support Vector Regression", fontsize=14)
# plt.savefig("SVR result", bbox_inches='tight', pad_inches=0)
# plt.show()


'''
kNN Regression
'''
from sklearn.neighbors import KNeighborsRegressor

source_s = data_config.loc[:, ['user_height', 'user_weight', 'user_age', 'bestfit_angle_standard']]
source_r = data_config.loc[:, ['user_height', 'user_weight', 'user_age', 'bestfit_angle_relax']]

trainset_s, testset_s = train_test_split(source_s, test_size = 0.2)

for i in range(1, 30):
    model = KNeighborsRegressor(n_neighbors=i, weights="distance", metric="minkowski", metric_params={"p":4})
    model.fit(x_train, y_train)
    train_score.append(model.score(x_train, y_train))
    test_score.append(model.score(x_test, y_test))
    print("Running for ", i)