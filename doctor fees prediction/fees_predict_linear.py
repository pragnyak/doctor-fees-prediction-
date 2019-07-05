# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:33:12 2019

@author: prasad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_tr=pd.read_csv("X_tr1.csv")
X_train=X_tr.iloc[:,1:].values
y_tr=pd.read_csv("y_tr1.csv")
y_train=y_tr.iloc[:,1:].values
X_te=pd.read_csv("X_te1.csv")
X_test=X_te.iloc[:,1:].values

#applying multilinear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
#prediction
y_pred_lin=regressor.predict(X_test)

#converting prediction to csv
ypredlin=pd.DataFrame(y_pred_lin)
ypredlin.to_csv("predictionlinear.csv")


