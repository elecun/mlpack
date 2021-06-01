'''
@brief  Leg-Rest Pos Recommendataion with DecisionTree Regressor
@author Byunghun Hwang <bh.hwang@iae.re.kr>
@date   2021. 05. 21
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import progressbar



'''
Presets & Hyper-parameters
'''
CONFIGURATION_FILE_PATH = "./data/train/data_config.csv"
DATASET_PATH = "./data/train/"
pd.set_option('display.width', 200) # for display width
# FEATURE_LENGTH = 30 # n-dimensional data feature only use
# NUMBER_OF_SAMPLES = 299 # number of augmented data
# FEATURE_MAX_LENGTH = 115 # Maximum feature length
# NUMBER_OF_RANDOM_SELECTION = 5
# MAX_TRAIN_ITERATION = -1 # infinity



'''
1. Load configuration file
'''
data_config = pd.read_csv(CONFIGURATION_FILE_PATH, header=0, index_col=0)


'''
2. data extraction
'''
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
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(ys), test_size=0.33, shuffle=True)

print("------ Regression Model Evaluation (@standard) ------")
model_standard = DecisionTreeRegressor(
    criterion = "mse",
    max_depth=6, 
    min_samples_leaf=1, 
    random_state=1).fit(X_train, y_train)

print("* R2 Score with Trainset (@standard) :", model_standard.score(X_train, y_train))
print("* R2 Score with Testset (@standard) :", model_standard.score(X_test, y_test))
print("* Feature Impotances (@standard) :")
for name, value in zip(X_train.columns, model_standard.feature_importances_):
    print('  - {0}: {1:.3f}'.format(name, value))


print("------ Regression Model Evaluation (@relax) ------")
model_relax = DecisionTreeRegressor(
    criterion = "mse", # mean square error
    max_depth=6, 
    min_samples_leaf=1, 
    random_state=1).fit(X_train, y_train)

print("* R-squared Score with Trainset (@relax) :", model_relax.score(X_train, y_train))
print("* R-squared Score with Testset (@relax) :", model_relax.score(X_test, y_test))
print("* Feature Impotances (@relax) :")
for name, value in zip(X_train.columns, model_relax.feature_importances_):
    print('  - {0}: {1:.3f}'.format(name, value))

'''
Output File Generation
'''
# min_age = 20
# max_age = 80
# ages = np.array([min_age+i for i in range(max_age-min_age+1)])

ages = np.arange(20, 80, step=10)

# min_height = 150
# max_height = 190
# heights = np.array([min_height+i for i in range(max_height-min_height+1)])
heights = np.arange(150, 190, step=10)

# min_weight = 40
# max_weight = 100
# weights = np.array([min_weight+i for i in range(max_weight-min_weight+1)])
weights = np.arange(40, 100, step=10)

bar = progressbar.ProgressBar(maxval=len(ages)*len(heights)*len(weights), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
output_standard = pd.DataFrame(columns=['age','height','weight','legrest'])
output_relax = pd.DataFrame(columns=['age','height','weight','legrest'])
count = 0
for a in ages:
    for h in heights:
        for w in weights:
            bmr = 66.47+(13.75*w)+(5*h)-(6.76*a)
            bmi = w/(h/100*h/100)
            pvs = model_standard.predict([[a,h,w,bmr,bmi]])
            pvr = model_relax.predict([[a,h,w,bmr,bmi]])
            print("Predict Result : standard({}), relax({})".format(pvs[0], pvr[0]))
            output_standard = output_standard.append({'age':a, 'height':h, 'weight':w, 'legrest':pvs[0]}, ignore_index=True)
            output_relax = output_relax.append({'age':a, 'height':h, 'weight':w, 'legrest':pvr[0]}, ignore_index=True)
            count = count+1
            bar.update(count)
bar.finish()

output_standard.to_csv('result_standard.csv', index=False)
output_relax.to_csv('result_relax.csv', index=False)
print("saved results")