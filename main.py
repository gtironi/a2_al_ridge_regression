# useful imports
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from a2_al_ridge_regression.ridge_model import ridge_fit, ridge_predict

alpha = 1

global_temperatures = pd.read_csv('globaltemperature/GlobalTemperatures.csv', parse_dates=['dt']) #read the data from csv
mean_temp_year = global_temperatures.groupby(global_temperatures.dt.dt.year).mean().drop('dt', axis=1).reset_index().rename({'dt': 'year'}, axis = 1) #group the data by year and made some modifications in the columns

less_2000 = mean_temp_year[mean_temp_year.year < 2000] #filter the data. Take just data with year < 2000
train, test = train_test_split(less_2000, test_size=0.2, random_state=42) #split the dataset into train and test set

# more_2000 = mean_temp_year[mean_temp_year.year >= 2000]

predictors = "year" #set the column year as the predictor
target = "LandAverageTemperature" #set the column LandAverageTemperature as the thing we need to predict

temperatures = list(train[target]) #make a list with the data from column target
years_train = list(train[predictors]) #make a list with the data from column predict (train set)
years_test = list(test[predictors]) #make a list with the data from column predict (test set)

x_train = []
x_test = []
y_train = temperatures

for y in years_train:
    x_train.append([y]) #make a list of lists with year values
    
for y in years_test:
    x_test.append([y]) #make a list of lists with year values

reg = Ridge(alpha=alpha) #set the ridge regression from sklearn
reg.fit(x_train, y_train) #apply the ridge regression from sklearn
prediction_sklearn = reg.predict(x_test) #predict data of test set with ridge regression 

#linear algebra approach
B = ridge_fit(train, predictors, target, alpha)
predictions = ridge_predict(test, predictors, B)

#check the difference between sklearn and the linear algebra way that we implement
print(predictions - prediction_sklearn)


