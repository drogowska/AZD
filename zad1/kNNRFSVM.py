
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
algs = [KNeighborsClassifier(), RandomForestClassifier(), SVC()]
params = [
    {'n_neighbors': np.arange(1, 25)},
    {'n_estimators': [100, 200, 300]},
    {'C': [0.1, 1, 10, 100, 1000],  
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['rbf']}  
]
best_parm = []
scores = []
gss = []

# knn + rf + voting
def kNNRFSVM(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    for i in range(len(algs)):
        gs = GridSearchCV(algs[i], params[i], cv=5)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        scores.append(gs.score(X_test,y_test))
        best_parm.append(gs.best_estimator_)
        gss.append(gs)

    ensemble = VotingClassifier(estimators=[('knn', best_parm[0]), ('rf', best_parm[1]), ('svm', best_parm[2])],voting='hard')
    ensemble.fit(X_train, y_train)
    acc = ensemble.score(X_test, y_test)
    pred = ensemble.predict(X_test)
    
    print('knn: {}'.format(best_parm[0].score(X_test, y_test)) + ', params: {}'.format(gss[0].best_params_))
    print('rf: {}'.format(best_parm[1].score(X_test, y_test)) + ', params: {}'.format(gss[1].best_params_))
    print('svm: {}'.format(best_parm[2].score(X_test, y_test)) + ', params: {}'.format(gss[2].best_params_))

    sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot=True)
    plt.show()
    print(classification_report(y_test, pred))
