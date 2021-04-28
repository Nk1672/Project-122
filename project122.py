import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = fetch_openml('mnist_784',version = 1,return_X_y = True)
# print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'],
n_classes=len(classes)

sample_class = 26
figure = plt.figure(figsize = (n_classes*2,(1 + sample_class * 2)))
idx_class = 0

for cls in classes:
  idxs = np.flatnonzero(y==cls)
  idxs = np.random.choice(idxs, sample_class, replace = False)
  i = 0
  for id in idxs:
    plt_idx = i*n_classes + idx_class + 1
    img = plt.subplot(sample_class, n_classes, plt_idx)
    img = sb.heatmap(np.reshape(X[id],(28,28)), cmap = plt.cm.gray, xticklabels = False, yticklabels = False, cbar = False)
    img = plt.axis('off')
    i+=1
  idx_class+=1

# print(len(X))
# print(len(X[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
print('Accuracy: ',accuracy_score(y_test, y_pred))

cm = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
img = plt.figure(figsize = (10,10))
img = sb.heatmap(cm, annot = True, fmt = 'd', cbar = False)