
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
import numpy as np
from main import preprocessing0
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# knn + rf + voting

X_train, X_test, y_train, y_test = preprocessing0()
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
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

rf = RandomForestClassifier()
params_rf = {'n_estimators': [100, 200, 300]}
rf_gs = GridSearchCV(rf, params_rf, cv=5)
rf_gs.fit(X_train, y_train)
rf_best = rf_gs.best_estimator_
print(rf_gs.best_params_)

ensemble = VotingClassifier(estimators=[('knn', knn_best), ('rf', rf_best)],voting='hard')
ensemble.fit(X_train, y_train)
acc = ensemble.score(X_test, y_test)
pred = ensemble.predict(X_test)
print(acc)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))