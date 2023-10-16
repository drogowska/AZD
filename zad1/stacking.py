from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report

def knntreesvm(X_train, X_test, y_train, y_test):

    level0 = list()
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('tree', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    # define meta learner model
    level1 = SVC()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    models = dict()
    models['knn'] = KNeighborsClassifier()
    models['tree'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['stacking'] = model

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    results, names = list(), list()
    for name, model in models.items():
        scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        results.append(scores)
        names.append(name)
        print('%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()





