## Dimensionality Reduction
**LDA**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Wine.csv')

x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

**Splitting the dataset**

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

**Feature Scaling**

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

**Applying LDA**

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

x_train = lda.fit_transform(x_train, y_train) #include y_train due that is a supervised learning
x_test = lda.transform(x_test)

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
            alpha=0.25,cmap = ListedColormap(('red','green','blue')))
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
plt.show()
