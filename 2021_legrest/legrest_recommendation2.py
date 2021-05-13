



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
    print(msg)