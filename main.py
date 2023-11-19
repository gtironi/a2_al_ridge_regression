from data_handle import global_temperature_data
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from ridge_model import ridge_fit, ridge_predict

alpha = 1

predictors = "year" #set the column year as the predictor
target = "LandAverageTemperature" #set the column LandAverageTemperature as the thing we need to predict

test, train, x_test, x_train, y_train = global_temperature_data(predictors, target)

reg = Ridge(alpha=alpha) #set the ridge regression from sklearn
reg.fit(x_train, y_train) #apply the ridge regression from sklearn
prediction_sklearn = reg.predict(x_test) #predict data of test set with ridge regression 

#linear algebra approach
B = ridge_fit(train, predictors, target, alpha)
predictions = ridge_predict(test, predictors, B)

#check the difference between sklearn and the linear algebra way that I implement
# print(predictions - prediction_sklearn)

# Polynomial ------------------------------------------------------------------------------------------ 

poly_reg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=alpha)) #defines the model as a polynomial of degree 2 [PolynomialFeatures(2)]
poly_reg.fit(x_train, y_train) #apply the ridge regression from sklearn
poly_predictions = poly_reg.predict(x_test) #predict data of test set with ridge regression 

print(poly_predictions - prediction_sklearn)


