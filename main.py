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

# Linear ------------------------------------------------------------------------------------------ 

reg = Ridge(alpha=alpha) #set the ridge regression from sklearn
reg.fit(x_train, y_train) #apply the ridge regression from sklearn
sklearn_linear_predictions = reg.predict(x_test) #predict data of test set with ridge regression 

#linear algebra approach
B = ridge_fit(train, predictors, target, alpha)
linear_predictions = ridge_predict(test, predictors, B)

#check the difference between sklearn and the linear algebra way that I implement
# print(linear_predictions - sklearn_linear_predictions)

# Polynomial 2 degrees ------------------------------------------------------------------------------------------ 

poly_reg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=alpha)) #defines the model as a polynomial of degree 2 [PolynomialFeatures(2)]
poly_reg.fit(x_train, y_train) #apply the ridge regression from sklearn
poly_predictions = poly_reg.predict(x_test) #predict data of test set with ridge regression 

# print(poly_predictions - sklearn_linear_predictions)

# Polynomial 2 degrees alpha test ------------------------------------------------------------------------------------------ 

alphas = [10**i for i in range(-2, 5)] #set the values of alpha to test
poly_predictions_alphas = []
for alpha in alphas: #execute the model with each alpha
    model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    poly_predictions_alphas.append(model.predict(x_test))
    
# print([poly_predictions_alphas[i][0] for i in range(len(alphas))])

# Polynomial multiple degrees ------------------------------------------------------------------------------------------ 

degrees = [1, 2, 4, 6]
multi_poly_predictions = []

for i in degrees:
    model = make_pipeline(PolynomialFeatures(i), Ridge(alpha=1))
    model.fit(x_train, y_train)
    multi_poly_predictions.append(model.predict(x_test))
    
print([multi_poly_predictions[i][0] for i in range(len(degrees))])