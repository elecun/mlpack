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
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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
source = data_config.loc[:, ['user_weight','user_height']]
bmi = source['user_weight']/(source['user_height']/100*source['user_height']/100)
target = data_config.loc[:, ['bestfit_angle_standard']]
target["bmi"] = bmi

corr = target.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(target['bmi'], target['bestfit_angle_standard']))

target = data_config.loc[:, ['bestfit_angle_relax']]
target["bmi"] = bmi

corr = target.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(target['bmi'], target['bestfit_angle_relax']))


# bmr
source = data_config.loc[:, ['user_weight','user_height', 'user_age']]
bmr = 66.47+(13.75*source['user_weight'])+(5*source['user_height'])-(6.76*source['user_age'])
target = data_config.loc[:, ['bestfit_angle_standard']]
target["bmr"] = bmr

corr = target.corr(method='pearson')
print('Pearson Correlation Coeff. (standard)\n', corr)
print('Psearson Correlation :', stats.pearsonr(target['bmr'], target['bestfit_angle_standard']))

target = data_config.loc[:, ['bestfit_angle_relax']]
target["bmr"] = bmr

corr = target.corr(method='pearson')
print('Pearson Correlation Coeff. (relax)\n', corr)
print('Psearson Correlation :', stats.pearsonr(target['bmr'], target['bestfit_angle_relax']))



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = data_config.loc[:, ['user_height', 'user_weight', 'user_age']]
bmr = 66.47+(13.75*X['user_weight'])+(5*X['user_height'])-(6.76*X['user_age'])
bmi = X['user_weight']/(X['user_height']/100*X['user_height']/100)
X['bmr'] = bmr
X['bmi'] = bmi
y = data_config.loc[:, ['bestfit_angle_standard']]


X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.2, random_state=42)

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(cv_results)
    print(msg)