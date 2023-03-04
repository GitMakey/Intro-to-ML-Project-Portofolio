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

# Basic summary statistics of data
means = np.round(X.mean(axis=0), 2)
stds = np.round(X.std(ddof=1, axis=0), 2)
medians = np.round(np.median(X, axis=0), 2)
ranges = np.zeros(X.shape[1])
for column in range(X.shape[1]):
    ranges[column] = np.round(X[:,column].max() -  (X[:,column].min()), 2)

summary_statistics = np.array([attributeNames, means, stds, medians, ranges])

# Simpler way of basic stats
stat = df.describe()
# Build table of statistics
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
stat_table = ax.table(cellText=np.round(stat.values, 2), colLabels=stat.columns, rowLabels=stat.index, loc='center')
stat_table.auto_set_font_size(False)
stat_table.set_fontsize(8)
stat_table.scale(1.5, 1)
#fig.tight_layout()
plt.savefig(os.path.join("plots","table.png"), dpi=200, bbox_inches='tight')
plt.show()

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
V = Vh.T

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
    plt.bar(r+i*bw, Vh[:,i], width=bw)
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
plt.plot(Z[:, i], Z[:, j], '.', alpha=.5)
        
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection PC1 - PC2')
plt.axis('equal')
plt.savefig(os.path.join("plots","projections.png"))
plt.show()

# Plot the first two PC  based on CHAS attribute

y_CHAS = X[:,3]
classNames_CHAS = ['Far from river','Tract bounds the river' ]
i = 0
j = 1
# Compute the projection onto the principal components
Z = U*S
# Plot projection
C = len(classNames_CHAS)
for c in range(C):
    plt.plot(Z[y_CHAS==c,i], Z[y_CHAS==c,j], '.', alpha=.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection PC1 - PC2')
plt.legend(classNames_CHAS)
plt.axis('equal')
plt.savefig(os.path.join('plots','projections CHAS.png'))
plt.show()

# Plot the first two PC  based on RM attribute

y_RM = X[:,5].round(0)
classNames_RM = ['4 Rooms','5 Rooms','6 Rooms','7 Rooms','8 Rooms','9 Rooms']
i = 0
j = 1
# Compute the projection onto the principal components
Z = U*S
# Plot projection
C = len(classNames_RM)
for c in range(C):
    plt.plot(Z[y_RM==c+4,i], Z[y_RM==c+4,j], '.', alpha=.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection PC1 - PC2')
plt.legend(classNames_RM)
plt.axis('equal')
plt.savefig(os.path.join('plots','projections RM.png'))
plt.show()


# Plot the histogram
nbins = 20
mu = 0
variance = 1
sigma = math.sqrt(variance)
fig, axs = plt.subplots(6, 2, figsize=(13,11))
fig.tight_layout(pad=3.0)
row = 0
col = 0
binary_counter = 0
for attr in range(X.shape[1]):
    if attributeNames[attr] in ["CHAS", "MEDV"]: 
        binary_counter += 1
    else:
        if binary_counter > 0:
            temp = attr - binary_counter
        else:
            temp = attr
            
        row = temp % 6
        col = temp // 6
        Xtemp = Y[:,attr]
        axs[row, col].set_title(f'{attributeNames[attr]}')
        axs[row, col].hist(Xtemp, bins=nbins, density=True)

        # Over the histogram, plot the theoretical probability distribution function:
        x = np.linspace(Xtemp.min(), Xtemp.max(), 1000)
        pdf = stats.norm.pdf(x,loc=mu,scale=sigma)
        axs[row, col].plot(x, pdf,'.',color='red', linewidth=1.0)
fig.savefig(os.path.join('plots','Normal distribution.png'))
plt.show()

# Plot boxplot of original data
plt.boxplot(X)
plt.xticks(r+bw, attributeNames, rotation=45)
plt.ylabel('value')
plt.title('Housing values in suburbs of Boston: Attribute values - non stadarized')
plt.savefig(os.path.join('plots','boxplot_non_stadarized.png'))
plt.show()

# Plot boxplots of standarized data
plt.boxplot(zscore(X, ddof=1), list(attributeNames))
plt.xticks(r+bw, attributeNames, rotation=45)
plt.ylabel('value')
plt.title('Housing values in suburbs of Boston: Standardized attribute values')
plt.savefig(os.path.join('plots','boxplot_standarized.png'))
plt.show()

#Plot correlation of all attributes
cmap = "Blues"
plt.figure(figsize=(12, 9))
sns.heatmap(np.corrcoef(Y.T),annot = True, fmt='.2g', square = True, cmap = cmap)#"YlOrBr")
plt.xticks(ticks=np.arange(X.shape[1])+.5, labels=attributeNames, rotation = 90)
plt.yticks(ticks=np.arange(X.shape[1])+.5, labels=attributeNames, rotation = 90)
plt.title('Correlation Matrix Heatmap')
plt.savefig(os.path.join('plots','heatmap.png'))




 


