import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import model_selection
from toolbox_02450 import *
import torch
from data import *

def standardize(X):

    mu_X = np.mean(X, 0)
    sigma_X = np.std(X, 0)

    X = (X - mu_X ) / sigma_X
    
    return X


def error_fn(y_hat, y):
   
    error = np.square(y-y_hat).sum(axis=0)/y.shape[0]
    return error

def mse(y_hat, y):
    return np.mean((y - y_hat) ** 2)

# Baseline for regression
class BaseLine_Regression:
    def fit(self, X, y):
        self.mean_ytrain = y.mean()

    def predict(self, X):
        return self.mean_ytrain
    
    class NeuralNetwork():
    

        def __init__(self, M, hidden_units):
        
        
            # Define the model
            self.model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            self.loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
        def fit(self,X,y):      
    
            # Extract training and test set for current CV fold, 
            # and convert them to PyTorch tensors
            self.X_nn = torch.Tensor(X)
            self.y_nn = torch.Tensor(y)
            self.new_shape = (len(y), 1)
            self.y_nn = self.y_nn.view(self.new_shape)
            
            self.net, self.final_loss, self.learning_curve = train_neural_net(self.model,
                                                        self.loss_fn,
                                                        X=self.X_nn,
                                                        y=self.y_nn,
                                                        n_replicates=1,
                                                        max_iter=10000)
      
        def predict(self,X):   
            
            self.X_nn = torch.Tensor(X)
            
            y_test_est = self.net(self.X_nn)
            return y_test_est
        
# Create two Layer Cross Validation function
def twoLayerCrossValidation(model_types, parameter_types, X, y, K1, K2):
    # Set K-folded CV options
    random_seed = 1234  # Set random seed to ensure same results whenever CV is performed

    K1fold = model_selection.KFold(n_splits=K1, shuffle=True, random_state=random_seed)
    K2fold = model_selection.KFold(n_splits=K2, shuffle=True, random_state=random_seed)

    test_errors = np.zeros((K1, len(model_types) * 2))
    
    y_hat_list = []
    y_test_list = []
    for i in range(len(model_types)):
        y_hat_list.append([])

    # The two layer cross-validation algorithm
    for i, (par_index, test_index) in enumerate(K1fold.split(X, y)):
        print("Outer fold {} of {}".format(i + 1, K1))

        # Saves D_par and D_test to allow later statistical evaluation
        X_par = X[par_index, :]
        y_par = y[par_index]

        X_test = X[test_index, :]
        X_test = standardize(X_test)  ###st
        y_test = y[test_index]
        
        
        y_test_list.append(y_test)
        

        # Iterate over the three methods
        for m, (model_type, models) in enumerate(model_types):
            val_errors = np.zeros((K2, len(models)))

            # Inner cross validation loop
            for j, (train_index, val_index) in enumerate(K2fold.split(X_par, y_par)):
                X_train = X_par[train_index, :]
                y_train = y_par[train_index]
                X_train = standardize(X_train)   ###st

                X_val = X_par[val_index, :]
                y_val = y_par[val_index]
                X_val = standardize(X_val)   ###st

                # Test model type and calculate validation error for each model of the three methods
                for k, (name, parameter, model) in enumerate(models):
                    model.fit(X_train, y_train.squeeze())

                    y_hat = model.predict(X_val)
                    
                    if model_type == 'ANN':
                        y_hat = y_hat.type(torch.float).data.numpy()
                    
                    val_errors[j, k] = mse(y_hat, y_val)

            # Finds the optimal model
            inner_gen_errors = val_errors.sum(axis=0)
            best_model_index = np.argmin(inner_gen_errors)
            
            # Determine optimal model
            best_model_name, best_model_parameter, best_model = models[best_model_index]  

            if model_type == 'Base Line':
                X_par_st = standardize(X_par)       ###st            
                #model.fit(X_par, y_par.squeeze())
                model.fit(X_par_st, y_par.squeeze())
                y_hat = np.ones(len(y_test)) * model.predict(X_test)
            else:
                X_par_st = standardize(X_par) 
                #best_model.fit(X_par, y_par)
                best_model.fit(X_par_st, y_par)
                y_hat = best_model.predict(X_test)
                if model_type == 'ANN':
                        y_hat = y_hat.type(torch.float).data.numpy()
                        
            y_hat_list[m].append(y_hat.squeeze())

            test_errors[i, m * 2 + 1] = mse(y_hat, y_test)  # Lists test_errors for each method and each outer fold
            test_errors[i, m * 2] = best_model_parameter  # List the best parameter type belonging to test-error

    test_errors_folds = pd.DataFrame.from_records(data=test_errors,
                                                  columns=sum([[parameter_types[i], model_types[i][0]] for i in
                                                                range(len(model_types))], []))

    return test_errors_folds, y_hat_list, y_test_list

def RegressionStatistics(test_errors, hats, tests):
    print(test_errors)
    predicted_lin = np.concatenate(hats[0])
    predicted_ANN = np.concatenate(hats[1])
    predicted_BL = np.concatenate(hats[2])
    true_class = np.concatenate(tests)
    alpha = 0.05

    z_1, CI_lin_vs_ANN, p_lin_vs_ANN = mcnemar(true_class, predicted_lin, predicted_ANN, alpha = 0.05)
    z_2, CI_lin_vs_BL, p_lin_vs_BL = mcnemar(true_class, predicted_lin, predicted_BL, alpha = 0.05)
    z_3, CI_ANN_vs_BL, p_ANN_vs_BL = mcnemar(true_class, predicted_ANN, predicted_BL, alpha = 0.05)

    print("P_value for the null hypothesis: Lin = ANN: ",p_lin_vs_ANN)
    print(1-alpha, "% Confidence interval for difference in accuracy between lin and ANN: ", CI_lin_vs_ANN)
    print("")
    print("P_value for the null hypothesis: Lin = BL: ",p_lin_vs_BL)
    print(1-alpha, "% Confidence interval for difference in accuracy between lin and BL: ", CI_lin_vs_BL)
    print("")
    print("P_value for the null hypothesis: ANN = BL: ",p_ANN_vs_BL)
    print(1-alpha, "% Confidence interval for difference in accuracy between ANN and BL ", CI_ANN_vs_BL)





def RegressionStatistics2(models, X, y, K1):    #(test_errors, hats, tests):
    ### Statistical Test Evaluation (SETUP II)

    ## Statistical test settings
    random_seed = 1234
    loss_in_r_function = 2 ## This implies the loss is squared in the r_j formula of box 11.4.1 
    r_baseline_vs_linear  = []                 ## The list to keep the r test size 
    r_baseline_vs_ANN = []                 ## The list to keep the r test size 
    r_ANN_vs_linear = []                 ## The list to keep the r test size 
    alpha_t_test            = 0.05
    rho_t_test              = 1/K1
    CV_setup_ii = model_selection.KFold(n_splits=K1,shuffle=True, random_state = random_seed + 1) ## Ensures that the CV for setup ii test is never the same randomization as for the estimation CVs

  #  most_common_lambda  = 1000  #most_common_lambda = stats.mode(optimal_regularization_param_linear).mode[0].astype('float64')  
    y_true = []
    y_hat = []
        
    for i, (train_index, test_index) in enumerate(CV_setup_ii.split(X)):
            print('Computing setup II CV K-fold: {0}/{1}..'.format(i + 1, K1))

            
            # Saves D_par and D_test to allow later statistical evaluation
            X_train = X[train_index, :]
            X_train = standardize(X_train)  ###st
            y_train = y[train_index]

            X_test = X[test_index, :]
            X_test = standardize(X_test)  ###st
            y_test = y[test_index]
        
    
            
            for (m, model) in enumerate(models):
                    print(m, model)
                    #y_hat = model.predict(X_train)
                    if m == 0:
                        model.fit(X_train, y_train)                    
                        y_hat_linear = model.predict(X_test).reshape(-1,1)
                    if m == 1:   #ANN
                        model.fit(X_train, y_train)
                        y_hat_ANN = model.predict(X_test)                       
                        y_hat_ANN = y_hat_ANN.detach().numpy()
                    if m == 2:
                        model.fit(X_train, y_train.squeeze())
                        y_hat_baseline = np.ones((y_test.shape[0],1)) * model.predict(X_test).squeeze()
                
            
            ## Add true classes and store estimated classes    
            y_true.append(y_test)
            print(y_hat_linear, y_hat_ANN, y_hat_baseline)
            y_hat.append(np.concatenate([y_hat_linear, y_hat_ANN, y_hat_baseline], axis=1) )
            
            ## Compute the r test size and store it
            r_baseline_vs_linear.append( np.mean( np.abs( y_hat_baseline-y_test ) ** loss_in_r_function - np.abs( y_hat_linear-y_test) ** loss_in_r_function ) )
            r_baseline_vs_ANN.append( np.mean( np.abs( y_hat_baseline-y_test ) ** loss_in_r_function - np.abs( y_hat_ANN-y_test) ** loss_in_r_function ) )
            r_ANN_vs_linear.append( np.mean( np.abs( y_hat_ANN-y_test ) ** loss_in_r_function - np.abs( y_hat_linear-y_test) ** loss_in_r_function ) )

    
    ## Baseline vs linear regression    
    p_setupII_base_vs_linear, CI_setupII_base_vs_linear = correlated_ttest(r_baseline_vs_linear, rho_t_test, alpha=alpha_t_test)
        
    ## Baseline vs ANN   
    p_setupII_base_vs_ANN, CI_setupII_base_vs_ANN = correlated_ttest(r_baseline_vs_ANN, rho_t_test, alpha=alpha_t_test)
        
    ## Linear regression vs ANN   
    p_setupII_ANN_vs_linear, CI_setupII_ANN_vs_linear = correlated_ttest(r_ANN_vs_linear, rho_t_test, alpha=alpha_t_test)
    
        # ## Create output table for statistic tests
        # df_output_table_statistics = pd.DataFrame(np.ones((3,5)), columns = ['H_0','p_value','CI_lower','CI_upper','conclusion'])
        # df_output_table_statistics[['H_0']] = ['err_baseline-err_linear=0','err_ANN-err_linear=0','err_baseline-err_ANN=0']
        # df_output_table_statistics[['p_value']]         = [p_setupII_base_vs_linear,p_setupII_ANN_vs_linear,p_setupII_base_vs_ANN]
        # df_output_table_statistics[['CI_lower']]        = [CI_setupII_base_vs_linear[0],CI_setupII_ANN_vs_linear[0],CI_setupII_base_vs_ANN[0]]
        # df_output_table_statistics[['CI_upper']]        = [CI_setupII_base_vs_linear[1],CI_setupII_ANN_vs_linear[1],CI_setupII_base_vs_ANN[1]]
        # rejected_null                                   = (df_output_table_statistics.loc[:,'p_value']<alpha_t_test)
        # df_output_table_statistics.loc[rejected_null,'conclusion']   = 'H_0 rejected'
        # df_output_table_statistics.loc[~rejected_null,'conclusion']  = 'H_0 not rejected'
        # df_output_table_statistics                      = df_output_table_statistics.set_index('H_0')
        
        # ## Export df as csv
        # df_output_table_statistics.to_csv('Regression_statistic_test_50000_10.csv',encoding='UTF-8')
        
        
    print("P_value for the null hypothesis: Lin = ANN: ",p_setupII_ANN_vs_linear)
    print(1-alpha_t_test, "% Confidence interval for difference in accuracy between lin and ANN: ", CI_setupII_ANN_vs_linear)
    print("")
    print("P_value for the null hypothesis: Lin = BL: ",p_setupII_base_vs_linear)
    print(1-alpha_t_test, "% Confidence interval for difference in accuracy between lin and BL: ", CI_setupII_base_vs_linear)
    print("")
    print("P_value for the null hypothesis: ANN = BL: ",p_setupII_base_vs_ANN)
    print(1-alpha_t_test, "% Confidence interval for difference in accuracy between ANN and BL ", CI_setupII_base_vs_ANN)     