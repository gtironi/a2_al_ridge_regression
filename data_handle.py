import pandas as pd
from sklearn.model_selection import train_test_split

def global_temperature_data(predictors, target):
    global_temperatures = pd.read_csv('globaltemperature/GlobalTemperatures.csv', parse_dates=['dt']) #read the data from csv
    mean_temp_year = global_temperatures.groupby(global_temperatures.dt.dt.year).mean().drop('dt', axis=1).reset_index().rename({'dt': 'year'}, axis = 1) #group the data by year and made some modifications in the columns
    mean_temp_year = mean_temp_year[mean_temp_year.LandAverageTemperatureUncertainty <= 1] #filter out entries with uncertainty > 1 degree celsius

    less_2000 = mean_temp_year[mean_temp_year.year < 2000] #filter the data. Take just data with year < 2000
    train, test = train_test_split(less_2000, test_size=0.2, random_state=42) #split the dataset into train and test set

    # more_2000 = mean_temp_year[mean_temp_year.year >= 2000]

    y_train = list(train[target]) #make a list with the data from column target
    years_train = list(train[predictors]) #make a list with the data from column predict (train set)
    years_test = list(test[predictors]) #make a list with the data from column predict (test set)

    x_train = []
    x_test = []

    for y in years_train:
        x_train.append([y]) #make a list of lists with year values
        
    for y in years_test:
        x_test.append([y]) #make a list of lists with year values
        
    return test, train, x_test, x_train, y_train