## Dimensionality Reduction
**PCA**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Wine.csv')

x = data.iloc[:,0:13].values
y = data.iloc[:,13]

**Splitting the dataset**

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

**Feature Scaling**

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

**Applying PCA to dataset**

from sklearn.decomposition import PCA

pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

#to explain percentage of variance, usually choose the ones that explain more variance

explained_variance 

**Reapply pca with 2 components, the ones that explain more variance**

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

explained_variance 

**Fitting Logistic Regression to training set with dimensions reduced**

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

**Metrics**

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

**Visualizing the Training set results**

plt.figure(figsize=(10,5))
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.5,cmap = ListedColormap(('red','green','blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j, 1],
               c = ListedColormap(('red','green','blue'))(i), label=j)

plt.legend()
plt.show()

**Visualizing test results**


plt.figure(figsize=(10,5))
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.25,cmap = ListedColormap(('red','green','blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j, 1],
               c = ListedColormap(('red','green','blue'))(i), label=j)

plt.legend()

## PCA Analysis with a different dataset
## The Data

Let's work with the cancer data set again since it had so many features.

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()

print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])

df.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

Now we can transform this data to its first 2 principal components.

x_pca = pca.transform(scaled_data)

scaled_data.shape

x_pca.shape

Great! We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

pca.components_

In this numpy matrix array, each row represents a principal component, and each column relates back to the original features. we can visualize this relationship with a heatmap:

df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
