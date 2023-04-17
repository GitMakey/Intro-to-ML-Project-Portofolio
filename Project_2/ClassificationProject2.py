import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.stats import zscore
from scipy import stats
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from matplotlib.pyplot import (figure, plot,subplot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show, clim)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import feature_selector_lr, bmplot,rocplot, confmatplot

# Data process

# Load the Real-Estate csv data using the Pandas library
# URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/898072/1523358/data.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230301%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230301T160235Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1f4f05b433874186e8dea74d05881585e7bb23da6802a1a7d67bb61749fd784cfc845b318b1e35dd3e07a3d1fec99b1ad2b7f05d5eac6d584e9cfad6d5a6d2d414a259851464b32ad58db293bdb0c913a206a915a87155383f98ec1acb5f098aff7eb1a5f66ef10c1565caa6c8da14ed91c8f964235f216f595f6175cf279ed8e792e093f19b1aa4223d21bf404d6055a4f2b06a51af27513c10683662a56669d04d1fa0cbb5dc76b01f46c0af2ba27a7c31a018a3111c0eeb81dadeff2ebf87adc770e5bc7a28ca998c79d7c356b99cb3b93e2309c806b8ef6c8218a3290d33fddacfae95bf42308b0d00c22e1df9f1c54b94a80136f1aad8eec8d3e3459aeb'
df = pd.read_csv('real_estate_dataset.csv')

# Check null
df.isnull().any()

# Check the number of null then decide how to handle missing data
df['RM'].isnull().sum()

# Drop the missing data
df = df.dropna()

# Extract attribute names
columns = list(df.columns)
attributeNames = np.asarray(columns)

# Convert the dataframe to numpy arrays
X = df.values

# Number of data objects and number of attributes
N, M = X.shape 

X = df.drop(columns=['RM'])
y = df['RM'].squeeze()
#y = df['RM'].astype(int).squeeze()
# transform y data to binary (1:Big, 0:Small)
# use numpy's where function to apply the discretization
y_discretized = np.where(y > 6, 1, 0)


#C = df['C'][0,0]
#M = df['M'][0,0]
#N = df['N'][0,0]

#attributeNames = [i[0][0] for i in df['attributeNames']]
#classNames = [j[0] for i in df['classNames'] for j in i]


# Remove outliers
outlier_mask = (X['CRIM']>30) | (X['B']<200)
outlier_mask = np.array(outlier_mask)
valid_mask = np.logical_not(outlier_mask)
X = np.array(X)
X = X[valid_mask,:]
y = np.array(y_discretized)
y = y[valid_mask]

# Update N and M
N, M = X.shape

print('Finish data process')

# Classification Part no2:

# Logistic Regression with regularization parameter
C = 1.0

model = lm.LogisticRegression(penalty='l2', C=C, solver='lbfgs')
model = model.fit(X,y)

# Classify houses as Big/Small(0/1) and assess probabilities
y_est_lr = model.predict(X)
y_est_big_prob = model.predict_proba(X)[:, 0] 

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est_lr != y) / float(len(y_est_lr))

# Display classification results
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_big_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_big_prob[class1_ids], '.r')
xlabel('Data object (Houses)'); ylabel('Predicted  prob. of class Big');
legend(['Big', 'Small'])
ylim(-0.01,1.5)

show()

print('Logistic Regression')

# Method 2: KNN

# Load data file and extract variables of interest
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
C = 2

# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


# K-nearest neighbors
K=5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=1

# To use a mahalonobis distance, we need to input the covariance matrix, too:
metric='mahalanobis'
metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est_knn = knclassifier.predict(X_test)


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est_knn==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est_knn);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()

print('KNN Classification')

# Baseline

# Compute the largest class in the training data
largest_class = np.argmax(np.bincount(y_train))

# Predict everything in the test data as belonging to the largest class
y_est_base = np.ones(y_test.shape, dtype=int) * largest_class

# Compute accuracy of the baseline model
accuracy_base = 100*np.sum(y_est_base == y_test)/len(y_test)
error_rate_base = 100 - accuracy_base

# Plot the classification results of the baseline model
figure(3)
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est_base == c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - Baseline');

# Compute and plot confusion matrix of the baseline model
cm_base = confusion_matrix(y_test, y_est_base);
accuracy_base = 100*cm_base.diagonal().sum()/cm_base.sum(); 
error_rate_base = 100-accuracy_base;
figure(4);
imshow(cm_base, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix of Baseline model (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy_base, error_rate_base));

show()

print('Baseline model classification')

# Two level cross validation

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LogisticRegression(fit_intercept=True)
    m = m.fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) == 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LogisticRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    

        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
         
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')


    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print('\n')
print('Logistic regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Logistic with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

# =============================================================================
# figure(k)
# subplot(1,3,2)
# bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
# clim(-1.5,0)
# xlabel('Crossvalidation fold')
# ylabel('Attribute')
# =============================================================================


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

# =============================================================================
# f=2 # cross-validation fold to inspect
# ff=Features[:,f-1].nonzero()[0]
# if len(ff) == 0:
#     print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
# else:
#     m = lm.LogisticRegression(fit_intercept=True).fit(X[:,ff], y)
#     
#     y_est= m.predict(X[:,ff])
#     residual=y-y_est
#     
#     figure(k+1, figsize=(12,6))
#     title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
#     for i in range(0,len(ff)):
#        subplot(2, int( np.ceil(len(ff) // 2)),i+1)
#        plot(X[:,ff[i]],residual,'.')
#        xlabel(attributeNames[ff[i]])
#        ylabel('residual error')
# =============================================================================
    
    
show()


print('Ran Two level cross validation for logistic regression')

## Crossvalidation for KNN
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
    
    # K-nearest neighbors
    neighbors = 5

    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=1

    # To use a mahalonobis distance, we need to input the covariance matrix, too:
    metric_m='mahalanobis'
    metric_m_params={'V': cov(X_train, rowvar=False)}

    # Compute squared error with all features selected (no feature selection)
    # Fit classifier and classify the test points
    m = KNeighborsClassifier(n_neighbors=neighbors, p=dist, 
                                        metric=metric_m,
                                        metric_params=metric_m_params)
    m.fit(X_train, y_train)
    
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
# =============================================================================
#     
#     Features[selected_features,k] = 1
#     # .. alternatively you could use module sklearn.feature_selection
#     if len(selected_features) == 0:
#         print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
#     else:
#         
#         # K-nearest neighbors
#         neighbors_f=5
# 
#         # Distance metric (corresponds to 2nd norm, euclidean distance).
#         # You can set dist=1 to obtain manhattan distance (cityblock distance).
#         dist=1
# 
#         # To use a mahalonobis distance, we need to input the covariance matrix, too:
#         metric_m='mahalanobis'
#         metric_m_params={'V': cov(X_train, rowvar=False)}
# 
#         # Fit classifier and classify the test points
#         knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
#                                             metric=metric_m,
#                                             metric_params=metric_m_params)
#         knclassifier.fit(X_train[:,selected_features], y_train)
#         
#         Error_train_fs[k] = np.square(y_train-knclassifier.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
#         Error_test_fs[k] = np.square(y_test-knclassifier.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
#     
#     
#     print('Cross validation fold {0}/{1}'.format(k+1,K))
#     print('Train indices: {0}'.format(train_index))
#     print('Test indices: {0}'.format(test_index))
#     print('Features no: {0}\n'.format(selected_features.size))
# 
#     k+=1
# 
# 
# # Display results
print('\n')
print('KNN without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('KNN with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
#     
#     
# show()
# 
# =============================================================================

print('Ran Two level cross validation for KNN')


