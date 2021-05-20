'''
@brief  Leg-Rest Pos Recommendataion with DecisionTree Regressor
@author Byunghun Hwang <bh.hwang@iae.re.kr>
@date   2021. 05. 21
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
from scipy import stats


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


'''
2. data extraction
'''
from sklearn.tree import DecisionTreeRegressor
X = data_config.loc[:, ['user_height', 'user_weight', 'user_age']]
bmr = 66.47+(13.75*X['user_weight'])+(5*X['user_height'])-(6.76*X['user_age'])
bmi = X['user_weight']/(X['user_height']/100*X['user_height']/100)
X["bmr"] = bmr
X["bmi"] = bmi
ys = data_config.loc[:, ['bestfit_angle_standard']]
yr = data_config.loc[:, ['bestfit_angle_relax']]


'''
DecisionTree Regression Model
'''
print("------ Regression Model Evaluation (@standard) ------")
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(ys), test_size=0.33, shuffle=False)
model = DecisionTreeRegressor(
    criterion = "mse",
    max_depth=6, 
    min_samples_leaf=1, 
    random_state=1).fit(X_train, y_train)

print("* R2 Score with Trainset (@standard) :", model.score(X_train, y_train))
print("* R2 Score with Testset (@standard) :", model.score(X_test, y_test))
print("* Feature Impotances (@standard) :")
for name, value in zip(X_train.columns, model.feature_importances_):
    print('  - {0}: {1:.3f}'.format(name, value))


print("------ Regression Model Evaluation (@relax) ------")
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(yr), test_size=0.33, shuffle=False)
model = DecisionTreeRegressor(
    criterion = "mse", # mean square error
    max_depth=6, 
    min_samples_leaf=1, 
    random_state=1).fit(X_train, y_train)

print("* R-squared Score with Trainset (@relax) :", model.score(X_train, y_train))
print("* R-squared Score with Testset (@relax) :", model.score(X_test, y_test))
print("* Feature Impotances (@standard) :")
for name, value in zip(X_train.columns, model.feature_importances_):
    print('  - {0}: {1:.3f}'.format(name, value))

'''
Output File Generation
'''
min_age = 10
max_age = 80
ages = np.array([min_age+i for i in range(max_age-min_age+1)])

min_height = 150
max_height = 200
heights = np.array([min_height+i for i in range(max_height-min_height+1)])

min_weight = 40
max_weight = 100
weights = np.array([min_weight+i for i in range(max_weight-min_weight+1)])

for a in ages:
    for h in heights:
        for w in weights:


