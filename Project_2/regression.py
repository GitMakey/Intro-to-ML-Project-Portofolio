import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd

from sklearn import model_selection
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

# Load the Real-Estate csv data using the Pandas library
# URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/898072/1523358/data.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230301%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230301T160235Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1f4f05b433874186e8dea74d05881585e7bb23da6802a1a7d67bb61749fd784cfc845b318b1e35dd3e07a3d1fec99b1ad2b7f05d5eac6d584e9cfad6d5a6d2d414a259851464b32ad58db293bdb0c913a206a915a87155383f98ec1acb5f098aff7eb1a5f66ef10c1565caa6c8da14ed91c8f964235f216f595f6175cf279ed8e792e093f19b1aa4223d21bf404d6055a4f2b06a51af27513c10683662a56669d04d1fa0cbb5dc76b01f46c0af2ba27a7c31a018a3111c0eeb81dadeff2ebf87adc770e5bc7a28ca998c79d7c356b99cb3b93e2309c806b8ef6c8218a3290d33fddacfae95bf42308b0d00c22e1df9f1c54b94a80136f1aad8eec8d3e3459aeb'

df = pd.read_csv('real_estate_dataset.csv')

# Drop the missing data
df = df.dropna()

df_reg = df.drop('MEDV', axis=1)
# Extract attribute names

attributeNames = np.asarray(df_reg.columns)

# Convert the dataframe to numpy arrays
X = df_reg.values

# Number of data objects and number of attributes
N_X, M_X = X.shape 

#Convert target feature to numpy array
y = df.MEDV.values

# Subtract the mean from the data and divide by the attribute standard deviation to obtain a standardized dataset:
X_= X - np.ones((N_X, 1))*X.mean(0)
X_ = X_*(1/np.std(X_,0))

# PCA by computing SVD of Y
U,S,Vh = svd(X_,full_matrices=False)
V = Vh.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9


plt.figure(figsize=(10,7))
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
#plt.savefig(os.path.join('plots','variance_explained_pca.png'))
plt.show()

pca_cor = pd.DataFrame(X_@V)
pca_cor['MEDV'] = y


# Lets look at the correlation matrix now.
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
sns.heatmap(pca_cor.corr(),annot=True)
plt.title('Principal components correlation matrix')

N = len(y)
M = len(attributeNames)

# Initialize lamdas
lambdas = np.power(10.,range(-2,8))

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-2,8))

#Initialize variables
#T = len(lambdas)
M = 6
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

# Add offset attribute
X_ = np.concatenate((np.ones(N).reshape(-1, 1), X_), axis=1)[:,:6]
attributeNames = [u'Offset']+attributeNames

k=0
for train_index, test_index in CV.split(X_,y):
    
    # extract training and test set for current CV fold
    X_train = X_[train_index]
    y_train = y[train_index]
    X_test = X_[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
   
    # Display the results for the last cross-validation folOpd
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
   
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()


    k+=1

show()

# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

attributeNames = ['offset', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
    

...

# TRANSFORM THE RESULTS BACK INTO THE ORIGINAL FEATURE SPACE
attributeNames = np.asarray(df_reg.columns)
#attributeNames = [u'Offset']+attributeNames
#attributeNames = np.insert(attributeNames, 0, 'offset')

# X_train_pca is the first 5 principal components obtained from PCA
X_train_pca = X_train[:, 1:6] # the first column (0) is unit column

# Model is the regularized linear regression model fitted on X_train_pca
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Inverse transform the principal components back into the original feature space
X_train_reconstructed = X_train_pca @ V[:, :5].T

# Compute the feature weights from the coefficients of the regularized linear regression model
feature_weights = model.coef_.dot(V[:, :5].T)

# Print the weights of the original features
print('Feature weights:')
for i, name in enumerate(attributeNames):
    print('{:>15} {:>15}'.format(name, np.round(feature_weights[i], 2)))



