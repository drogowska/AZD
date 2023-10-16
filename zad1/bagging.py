from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
# Load the dataset

def nbtree(X_train, X_test, y_train, y_test):
    # Create the base classifier
    tree = DecisionTreeClassifier()
    bayes = GaussianNB()
    # Number of base models (iterations)
    n_estimators = 10
    # Create the Bagging classifier
    bagging_classifier_tree = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators)
    bagging_classifier_bayes = BaggingClassifier(base_estimator=bayes, n_estimators=n_estimators)
    # Create the ensemble classifier with both the Bagged and Naive Bayes classifiers
    ensemble = VotingClassifier(estimators=[('bagged1', bagging_classifier_tree), ('bagged2', bagging_classifier_bayes)], voting='hard')
    # Train the Bagging classifier
    ensemble.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = ensemble.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot=True)
    plt.show()
    print(classification_report(y_test, y_pred))
    return y_pred