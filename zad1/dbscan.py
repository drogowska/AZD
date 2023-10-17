
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np


params = [
    {'eps': np.arange(0.2, 10)},
    {'min_samples': np.arange(2, 50)} 
]

def dbscan(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    db = DBSCAN() 
    gs = GridSearchCV(db, params, cv=5, scoring="accuracy")               
    gs.fit(X_train, y_train)
    print('dbscan: {}'.format(gs.best_estimator_[0].score(X_test, y_test)) + ', params: {}'.format(gs.best_params_))

    y_pred = gs.predict(X_test)
    sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot=True)
    plt.show()
    print(classification_report(y_test, y_pred))
    return y_pred


