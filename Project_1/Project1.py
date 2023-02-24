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
URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/898072/1523358/data.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230224T151948Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=615472271ee20824c225dcdd5bf1eab187e51cf994cf1cc3c5431838d2aa5b6d4d4a2dddc6ba7fe60d22326a3b634b9746a209b3b66845000c89b799b978dd6f442242fe0f3b0d9962eabf6c663fe2fb5d61a39680d4aac078ec2344a6a20d222777e41e648ca7dd64998071a288b2a49c66c4073711ae164fb597e9c8882227d58d3542d5fb4caf309623cb998ad53d8560d43c9302069fc750cbbbb5bf389603211b400d6986cb3d9fd24731d9d186d266beb22bdc548104c24c81057fd4764a30a3b672704ef4dc2bf035f762e56741c3706b3d8903fcc425d2df6dc8f41383b45e9a60334e890c87014531b151eeb2eb18cef403d048606d1f0abd79f251'
df = pd.read_csv(URL)

# Check null
df.isnull().any()

# Check the number of null then decide how to handle missing data
df['RM'].isnull().sum()

# Drop the missing data
df = df.dropna()

# Extract attribute names
columns = list(df.columns)
columns.remove('RM')
attributeNames = np.asarray(columns)



# Convert the dataframe to numpy arrays
raw_data = df.values

#Data matrix X
X1 = raw_data[:, 0:5]
X2 = raw_data[:, 6: ]
X = np.hstack((X1, X2))

#Data matrix y 
y = raw_data[:, 5]
# Transform to binary 
#y = np.where(y<=6, 0, 1) # More than 6 rooms=0, less=1
#classNames = ['Less than 6 rooms', 'More than three rooms']
classNames = ['4 rooms', '5 rooms', '6 rooms', '7 rooms', '8 rooms', '9 rooms']
patata = np.round(y, 0)


# Number of data objects and number of attributes
N, M = X.shape 

# Number of Classes 
C = np.unique(patata)
#C = len(np.unique(y))

# Basic summary statistics of data
means = np.round(X.mean(axis=0), 2)
stds = np.round(X.std(ddof=1, axis=0), 2)
medians = np.round(np.median(X, axis=0), 2)
ranges = np.zeros(X.shape[1])
for column in range(X.shape[1]):
    ranges[column] = np.round(X[:,column].max() -  (X[:,column].min()), 2)

summary_statistics = np.array([attributeNames, means, stds, medians, ranges])

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig(os.path.join('plots','variance_explained_pca.png'))
plt.show()

# Plot PCA Component Coefficients
pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.xticks(rotation=45)
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Housing values in suburbs of Boston: PCA Component Coefficients')
plt.savefig(os.path.join('plots','pca_component_coef.png'))
plt.show()

# Choose the first two PCs to plot (the projection)
i = 0
j = 1
# Compute the projection onto the principal components
Z = U*S
# Plot projection
for c in range(len(C)):
    plt.plot(Z[patata==C[c],i], Z[patata==C[c],j], '.', alpha=.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection PC1 - PC2')
plt.legend(classNames)
plt.axis('equal')
plt.savefig(os.path.join('plots','projections.png'))
plt.show()

# Plot the histogram
nbins = 20
mu = 0
variance = 1
sigma = math.sqrt(variance)
fig, axs = plt.subplots(7, 2, figsize=(13,11))
fig.tight_layout(pad=3.0)
col = 0
ro = 0

for attr in range(X.shape[1]):
    col = attr % 7
    ro = attr // 7
    Xtemp = Y[:,attr]
    axs[col, ro].set_title(f'{attributeNames[attr]}')
    axs[col, ro].hist(Xtemp, bins=nbins, density=True)

    # Over the histogram, plot the theoretical probability distribution function:
    x = np.linspace(Xtemp.min(), Xtemp.max(), 1000)
    pdf = stats.norm.pdf(x,loc=mu,scale=sigma)
    axs[col, ro].plot(x, pdf,'.',color='red', linewidth=1.0)
fig.savefig(os.path.join('plots','histograms.png'))
plt.show()

# Plot boxplot of original data
plt.boxplot(X)
plt.xticks(r+bw, attributeNames, rotation=45)
plt.ylabel('value')
plt.title('South African Heart Disease: Attribute values')
plt.savefig(os.path.join('plots','box.png'))
plt.show()

# Plot boxplots of standarized data
plt.boxplot(zscore(X, ddof=1), list(attributeNames))
plt.xticks(r+bw, attributeNames, rotation=45)
plt.ylabel('value')
plt.title('South African Heart Disease: Standardized attribute values')
plt.savefig(os.path.join('plots','box_stand.png'))
plt.show()

#Plot correlation of all attributes
cmap = "Blues"
plt.figure(figsize=(12, 9))
sns.heatmap(np.corrcoef(Y.T),annot = True, fmt='.2g', square = True, cmap = cmap)#"YlOrBr")
plt.xticks(ticks=np.arange(X.shape[1])+.5, labels=attributeNames, rotation = 90)
plt.yticks(ticks=np.arange(X.shape[1])+.5, labels=attributeNames, rotation = 90)
plt.title('Correlation Matrix Heatmap')
plt.savefig(os.path.join('plots','heatmap.png'))




 


