import pandas as pd
import numpy as np
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.stats import zscore
from scipy import stats

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

#Logistic Regression

X = df.drop(columns=['RM'])
y = df['RM'].astype(int).squeeze()
#C = df['C'][0,0]
#M = df['M'][0,0]
#N = df['N'][0,0]

#attributeNames = [i[0][0] for i in df['attributeNames']]
#classNames = [j[0] for i in df['classNames'] for j in i]


# Remove outliers
outlier_mask = (X['CRIM']>30) | (X['B']<200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]

# Update N and M
N, M = X.shape

print('Ran Exercise 5.1.5')

model = lm.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12]).reshape(1,-1)
# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being a white wine: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object (wine sample)'); ylabel('Predicted prob. of class White');
legend(['White', 'Red'])
ylim(-0.01,1.5)

show()

print('Ran Exercise 5.2.6')
