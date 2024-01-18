# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:42:17 2024

@author: supra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from pyopls import OPLS
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.metrics import accuracy_score, f1_score, recall_score,confusion_matrix,roc_curve,roc_auc_score, RocCurveDisplay, classification_report


data = pd.read_csv(r'C:\Users\supra\winequality(white)\winequality-white.csv', delimiter=';')
X = data.drop('quality', axis='columns')
y = data['quality']
y_unique = y.unique()

y_clf = y.apply(lambda x: 0 if x<7 else 1)

scaling = StandardScaler()
X_scaled = scaling.fit_transform(X)

RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=RANDOM_SEED)

model = [ LogisticRegression, RandomForestClassifier, DecisionTreeClassifier]

for i in range(len(model)):
    clf = model[i]()
    clf.fit(X_train, y_train)
    print('Score of:', model[i], 'is', clf.score(X_test, y_test))
    
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)
print('\n Score:', model1.score(X_test, y_test))
y_pred = model1.predict(X_test)

A = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(3,3))
sns.heatmap(A, annot=True, cbar=True)
plt.xlabel('True')
plt.ylabel('Predicted') 
   
report = classification_report(y_test, y_pred)
RocCurveDisplay.from_estimator(model1, X_test, y_test)
RocCurveDisplay.from_predictions(y_test, y_pred)   

# grid = { 'n_estimators' : [25,50,100,150],
#           'max_features' : ['sqrt', 'log2', None], 
#           'max_depth' : [3,6,9],
#           'max_leaf_nodes' : [3,6,9],
#           'criterion' : ['gini', 'entropy'],
#           # 'bootstrap' : [True, False]
#         }
# begin = time()
# RS_clf = RandomizedSearchCV(model1, grid, n_iter=200, cv=2, verbose=2, random_state=RANDOM_SEED,
#                             n_jobs=-1, error_score='raise')
# RS_clf.fit(X_train, y_train)
# print('\n Score afte hyper parameter tuning:',RS_clf.score(X_test, y_test))
# print('\n Best parametrs:', RS_clf.best_params_)
# end =time()
# print('\n Time taken for RandomizedSearchCV :', end-begin)
    
feature_significane = model1.feature_importances_
feature = X.columns
plt.figure(figsize=(12,5))
plt.bar(feature, feature_significane, width=0.5) 
plt.tight_layout()  
plt.show() 
    
def opls_func(n_components, X,y):
    opls = OPLS(n_components=1)
    Z =opls.fit_transform(X,y)
    pls = PLSRegression(1)
    pls.fit(Z,y)
    df = pd.DataFrame(np.column_stack([pls.x_scores_, opls.T_ortho_[:,0]]),
                      index=X.index,columns = ['x_score','t_ortho'])
    a = df[y==3]
    b = df[y==4]
    c = df[y==5]
    d = df[y==6]
    e = df[y==7]
    f = df[y==8]
    g = df[y==9]
    
    plt.scatter(a['x_score'], a['t_ortho'], c='red', label=3)
    plt.scatter(b['x_score'], b['t_ortho'], c='orange', label=4)
    plt.scatter(c['x_score'], c['t_ortho'], c='yellow', label=5)
    plt.scatter(d['x_score'], d['t_ortho'], c='blue', label=6)
    plt.scatter(e['x_score'], e['t_ortho'], c='lightgreen', label=7)
    plt.scatter(f['x_score'], f['t_ortho'], c='green', label=8)
    plt.scatter(g['x_score'], g['t_ortho'], c='black', label=9)
    
    plt.axhline(y=0, color='black', linestyle ='--')
    plt.axvline(x=0, color='black', linestyle ='--')
    plt.title('OPLS DA Analysis')
    plt.xlabel('x_score')
    plt.ylabel('t_ortho')
    plt.xlim(-10,20)
    plt.legend( bbox_to_anchor=(1,1), loc='center left')
    plt.show()
       

opls_func(1, X, y)





    
    
    
    
    
    
    
    
    
    
    
    