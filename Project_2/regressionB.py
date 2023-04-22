from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import math

#from models import *
from data import *
from functions_reg import *



K1 = 10
K2 = 10


# Regularized Linear Regression
_lambdas = np.logspace(-1, 2, 10)
#_lambdas = np.logspace(-4, 4, k)

# Neural Network
_hidden = np.arange(1,4)
#_hidden = np.arange(1,8)
M = 3 # number of features



model_types = [
    ("Ridge Regression", [
        ("Ridge Lambda = {}".format(_lambda), _lambda, linear_model.Ridge(alpha=_lambda, fit_intercept=True))
        for _lambda in _lambdas
    ]),
    ("ANN", [
        ("Hidden units = {}".format(h), h, NeuralNetwork(M, hidden_units=h))
        for h in _hidden
    ]),
    ("Base Line", [("Base Line", None, BaseLine_Regression())])

]



parameter_types = ["Lambda_vals", "Hidden Units", "None"]  # The names of the parameters, for later convenience



test_errors, hats, tests = twoLayerCrossValidation(model_types, parameter_types, X[:,1:], y, K1, K2)

optimal_lambda = test_errors.iloc[:,0].value_counts().idxmax()
optimal_h = int(test_errors.iloc[:,2].value_counts().idxmax())


models = [linear_model.Ridge(alpha=optimal_lambda, fit_intercept=True), NeuralNetwork(M, hidden_units=optimal_h), BaseLine_Regression()]

RegressionStatistics2(models, X[:,1:], y, K1)
