
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
import numpy as np
from main import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC


warnings.filterwarnings('ignore')

# knn + rf + voting

X_train, X_test, y_train, y_test = preprocessing()
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_gs = GridSearchCV(knn, params_knn, cv=5)                    # cross validation
knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_
print(knn_gs.best_params_)
y_pred = knn_gs.predict(X_test)
print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

rf = RandomForestClassifier()
params_rf = {'n_estimators': [100, 200, 300]}
rf_gs = GridSearchCV(rf, params_rf, cv=5)
rf_gs.fit(X_train, y_train)
rf_best = rf_gs.best_estimator_
rf_pred = rf_gs.predict(X_test)
print(rf_gs.best_params_)
print(accuracy_score(y_test, rf_pred))

# svm =  make_pipeline(StandardScaler(), SVC(gamma='auto'))
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
svm_gs = GridSearchCV(SVC(), param_grid, cv=5) 
svm_gs.fit(X_train, y_train)
svm_best = svm_gs.best_estimator_
svm_pred = svm_gs.predict(X_test)
print(svm_best.best_params_)
print(svm_gs.score(X_test, y_test))



ensemble = VotingClassifier(estimators=[('knn', knn_best), ('rf', rf_best), ('svm', svm_best)],voting='hard')
ensemble.fit(X_train, y_train)
acc = ensemble.score(X_test, y_test)
pred = ensemble.predict(X_test)
print(acc)


print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('svm: {}'.format(svm_best.score(X_test, y_test)))

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))