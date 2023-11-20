from data_handle import global_temperature_data
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from ridge_model import ridge_fit, ridge_predict

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


alpha = 1

predictors = "year" #set the column year as the predictor
target = "LandAverageTemperature" #set the column LandAverageTemperature as the thing we need to predict

test, train, x_test, x_train, y_train, future_x, future_y = global_temperature_data(predictors, target)

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
    
# for i in range(len(alphas)): #print the results in the terminal
#     print(f'Alpha {alphas[i]}: {poly_predictions_alphas[i][0]}')

# Polynomial multiple degrees ------------------------------------------------------------------------------------------ 

degrees = [1, 2, 4, 6] #set the polynom degress to test
multi_poly_predictions = []

for i in degrees: #execute the model for each degree
    model = make_pipeline(PolynomialFeatures(i), Ridge(alpha=1)) 
    model.fit(x_train, y_train)
    multi_poly_predictions.append(model.predict(x_test))

# for i in range(len(degrees)): #print the results in the terminal
#     print(f'Degree {degrees[i]}: {multi_poly_predictions[i][0]}')

# Polynomial multiple degrees alpha test ------------------------------------------------------------------------------------------

alphas = [10**i for i in range(0, 5)] #set the values of alpha to test
degrees = [1, 2, 4, 6]
multi_poly_predictions_alphas = []

for i in degrees: #execute the model for each degree
    for alpha in alphas: #execute the model for each alpha
        model = make_pipeline(PolynomialFeatures(i), Ridge(alpha=alpha)) 
        model.fit(x_train, y_train)
        multi_poly_predictions_alphas.append(model.predict(x_test))
    
# for i in range(len(degrees)): #print the results in the terminal
#     print('')
#     for j in range(len(alphas)):
#         print(f'Degree {degrees[i]} and Alpha {alphas[j]}: {multi_poly_predictions_alphas[i*j][0]}')

# Cross Validation ------------------------------------------------------------------------------------------
from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True) #create a generator for a 10-fold
score_model = make_pipeline(PolynomialFeatures(4), Ridge(alpha=1)) #defines the model as a polynomial of degree 4 and alpha 1
results = model_selection.cross_val_score(score_model, x_train, y_train, cv=kfold, scoring= 'neg_mean_absolute_error') #train the model and apply the 10-fold cross validation

# print(f"score_model {results.mean()}") 

# Optimizing alpha and degree ------------------------------------------------------------------------------------------

alphas = [10**i for i in range(0, 10)] #set the values of alpha to test
degrees = [1, 2, 3, 4, 5, 6, 7] #set of degrees to test
scores = []
best_alphas = []
best_scores = []

for i in degrees: #using RidgeCV to select the best alpha for each degree
        modelCV = make_pipeline(PolynomialFeatures(i), RidgeCV(alphas=alphas, cv = 10)) 
        modelCV.fit(x_train, y_train)
        
        score = modelCV.score(x_train, y_train)
        scores.append((i, score)) #stores the model score in a list
        best_score = modelCV.named_steps['ridgecv'].best_score_ #verify the best score for the best alpha
        best_scores.append((i, best_score)) #stores the model best score in a list
        best_alpha = modelCV.named_steps['ridgecv'].alpha_ #verify the best alpha
        best_alphas.append((i, best_alpha)) #stores the model best alpha in a list
        
# print(scores)
# print(best_alphas)
        
# testing the models ------------------------------------------------------------------------------------------

# 6 degress, alpha 1
model6D = make_pipeline(PolynomialFeatures(6), Ridge(alpha=100000000)) 
model6D.fit(x_train, y_train)
predictions_6D = model6D.predict(x_test)

# 4 degress, alpha 10000
model4D = make_pipeline(PolynomialFeatures(4), Ridge(alpha=1000)) 
model4D.fit(x_train, y_train)
predictions_4D = model4D.predict(x_test)

from sklearn.metrics import mean_absolute_error

print(f"O erro do modelo P4 foi: {mean_absolute_error(test[target], predictions_4D)} ºC \nO erro do modelo P6 foi: {mean_absolute_error(test[target], predictions_6D)} ºC")

# testing the best model on future data ------------------------------------------------------------------------------------------

predictions_6D_future = model6D.predict(future_x)

print(f"O erro nos dados depois de 2000 foi de: {mean_absolute_error(future_y, predictions_6D_future)}ºC")

print(f"A previsão do modelo para a temperatura em 2060 é de {round(model6D.predict([[2060]])[0], 5)}ºC")

