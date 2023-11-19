import numpy as np
import pandas as pd

def ridge_fit(train, predictors, target, alpha):
    '''
    Função responsável por determianar os coeficientes da reta

    Parameters
    ----------
    train: dataframe
        O dataframe completo referente a base de treino
    predictors: str
        o nome da coluna a ser usada como 'x', ou seja, para prever
    target: str
        o nome da coluna a ser usada como 'y', ou seja, o que será previsto
    alpha: float
        o valor de alpha a ser usado na penalização
        
    Returns
    ----------
    B: dataframe
        matrix 1xN que contém os coeficientes da regressão linear
    '''
    X = pd.DataFrame(train[predictors].copy())
    y = train[[target]].copy()
    
    X["intercept"] = 1
    X = X[["intercept"] + [predictors]]
    
    penalty = alpha * np.identity(X.shape[1])
    penalty[0][0] = 0
    
    B = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
    return B

def ridge_predict(test, predictors, B):
    '''
    Função responsável por determianar os coeficientes da reta

    Parameters
    ----------
    test: dataframe
        O dataframe completo referente a base de teste
    predictors: str
        o nome da coluna a ser usada como 'x', ou seja, para prever
    B: dataframe
        matrix 1xN que contém os coeficientes da regressão linear
        
    Returns
    ----------
    predictions: ndarray
        array com as predições para cada valor de x passado
    '''
    test_X = pd.DataFrame(test[predictors])
    test_X["intercept"] = 1
    test_X = test_X[['intercept'] + [predictors]]

    predictions = test_X.values @ B.values
    return predictions.ravel()