import pandas as pd
import numpy as np


# Load csv with data
df = pd.read_csv('real_estate_dataset.csv')
df
# Drop the missing data
df = df.dropna()

# Extract attribute names (except last column)
attributeNames_ = df.columns[:-1]

# Extract vector y, convert to NumPy array
y = np.array(df.MEDV)
#y = np.array(df.MEDV)

# Extract , feture values to NumPy array
X_ = np.array(df.iloc[:, [1, 5, 11]])

# Compute values of N, M.
N_ = len(y)
M_ = len(attributeNames_)

# Add offset attribute
X = np.concatenate((np.ones(N_).reshape(-1, 1), X_), axis=1)


# Chose features and get necessary matrixes
attributeNames = ['ZN', 'RM', 'B']
y_attr = ['MEDV']



# Compute values of N, M.
N = len(y)
M = len(attributeNames_)+1

