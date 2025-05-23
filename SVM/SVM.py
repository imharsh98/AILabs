# ==== Linear Support Vector Machine ====

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris virginica

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# --- Standardizing the features & Build a linear SVM classifier ---

svm_clf = Pipeline([
     ("scaler", StandardScaler()),
     ("linear_svc", LinearSVC(C=1, loss="hinge")),
  ])

svm_clf.fit(X, y)

svm_clf.predict([[5.5, 1.7]])
 

# ==================================================================================

# SVM using Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd_clf = Pipeline([
      ("scaler", StandardScaler()),
      ("sgd", SGDClassifier(loss="hinge", alpha=0.01)),
  ])  # alpha=1/C

sgd_clf.fit(X, y)

sgd_clf.predict([[5.5, 1.7]])


# ==================================================================================

# Reference code for linear SVM classification

# --- Loading the Iris dataset from scikit-learn ---

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris virginica


# --- Splitting data into 70% training and 30% test data ---

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# --- Standardizing the features ---

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# --- Build a linear SVM classifier ---

from sklearn.svm import LinearSVC
svm_clf = LinearSVC(C=1, loss="hinge")
svm_clf.fit(X_train,y_train)


# --- Test the SVM model ---
y_hat=svm_clf.predict(X_test)


# --- Google Drive Path Setting ---
import os
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Colab Notebooks/AIPR112') 
work_path = os.getcwd()
print(work_path)


# --- Plot the decision boundary ---

import matplotlib.pyplot as plt
from PlotClassification import plot_decision_regions

import numpy as np
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=svm_clf, test_idx=range(115, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# ==================================================================================

# Partial code for SVM using polynomial kernel

from sklearn.svm import SVC
svm_clf = SVC(kernel="poly", degree=3, C=5)
svm_clf.fit(X_train, y_train)


# ==================================================================================

# Partial code for SVM using Gaussian RBF kernel

from sklearn.svm import SVC
svm_clf = SVC(kernel="rbf", gamma=5, C=1)
svm_clf.fit(X_train, y_train)






