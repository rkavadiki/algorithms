from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from classifier.LogisticRegression import LogisticRegression
from classifier.Gradient_Descent import Gradient_Descent
import os
os.chdir("/Users/rajesh.kavadiki/Documents/Projects/personal/algorithms/algorithms")
%load_ext autoreload
%autoreload 2

titanic = pd.read_csv("/Users/rajesh.kavadiki/Downloads/titanic.csv")
ports = pd.get_dummies(titanic.Embarked, prefix='Embarked')
titanic = titanic.join(ports)
titanic.drop(['Embarked', 'Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
titanic.Sex = titanic.Sex.map({'male': 0, 'female': 1})
titanic.Age.fillna(titanic.Age.mean(), inplace=True)
titanic.head()

diab = pd.read_csv("/Users/rajesh.kavadiki/Downloads/diabetes.csv")

y = diab.target.copy()
x = diab.drop(['target'], axis=1)
x.head()
y.shape


lr.fit(x, y)
clf = LogisticRegression().fit(x, y)
ypred = clf.predict_proba(x)[:, 0]
ytrue = df.Survived.values
print(np.sum(ytrue*np.log(ypred)+(1-ytrue)*np.log(1-ypred)), roc_auc_score(ytrue, ypred))

np.unique(df.Survived.values)
df.Survived.sum()
print(__name__)

x.head()

print(os.getcwd())
