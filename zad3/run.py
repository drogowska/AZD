from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import os.path
import seaborn as sns
from FuzzyCMeans import FuzzyCMeansDetector
from FuzzyHierarchicalClustering import FuzzyOutlierDetection
from FuzzyKMeans import KMeansDetector
from typing import Any, Dict, List, Tuple
import numpy as np


def visu(file):
    # t-SNE Visualization
    df = pd.read_csv(path + file, on_bad_lines='skip')
    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    tsne_results = tsne.fit_transform(df.iloc[:, :-1])
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['label'] = df.iloc[:, -1:]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
    plt.show()


def preprocessing():
    if not os.path.isfile(path + "X_trainwine.csv"):
        for file in files:
            df = pd.read_csv(path + file, on_bad_lines='skip')
            if file == "iris.csv":
                label_e = LabelEncoder()
                df[df.columns[4]] =label_e.fit_transform(df.iloc[:,-1]) 
                df[df.columns[4]] = df[df.columns[4]] + 1

            scaler = StandardScaler()
            df_2 = df.iloc[:, :-1]
            X = pd.DataFrame(scaler.fit_transform(df_2))
            y= df.iloc[:, -1:]
            split=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=9)
            for train,test in split.split(X,y):     
                X_train = X.iloc[train]
                y_train = y.iloc[train]
                X_test = X.iloc[test]
                y_test = y.iloc[test]
           
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=48)
            X_train.to_csv(path + 'X_train' + file)
            X_test.to_csv(path + 'X_test' + file)
            y_train.to_csv(path + 'Y_train' + file)
            y_test.to_csv(path + 'Y_test' + file)
            X.to_csv(path + 'X' + file)
            y.to_csv(path + 'Y' + file)





def build_tuple(file) -> Tuple[np.ndarray,np.ndarray]:
    X = pd.read_csv(path + 'X' + file).values[:, 1:]
    y = pd.read_csv(path + 'Y' + file).values[:, 1:].squeeze().astype(np.int32)
    return X,y




path  = "./data/"
# , 
files = ["dermatology.csv", "wine.csv"]
kmeans_n_clusters=20
kmeans_fraction_threshold=0.03
cmeans_n_clusters=20
cmeans_fraction_threshold=0.03
hk_threshold_distance=0.1
hk_threshold_cluster=3
preprocessing()


for file in files:
    print("Baza danych : " + file)
    visu(file)
    X_train = pd.read_csv(path + 'X_train' + file).values[:,1:]
    X_test = pd.read_csv(path + 'X_test' + file).values[:,1:]
    y_train = pd.read_csv(path + 'Y_train' + file).values[:,1:]
    y_test = pd.read_csv(path + 'Y_test' + file).values[:,1:]
    X = pd.read_csv(path + 'X' + file).values[:, 1:]
    y = pd.read_csv(path + 'Y' + file).values[:, 1:]

    y_train = np.concatenate(y_train, axis=0).astype(int)
    y_test = np.concatenate(y_test, axis=0)

    print('kmeans')

    dataset = build_tuple(file)

    DETECTORS: Dict[str, Any] = {
        "cmeans": FuzzyCMeansDetector,
        "kmeans": KMeansDetector,
    }

    detector = DETECTORS["kmeans"](dataset,
                                   str(file.replace('.csv','')) + '_kmeans_n_clusters_' +
                                   str(str(kmeans_n_clusters).replace('.',',')) +
                                   '_fraction_threshold_' +
                                   str(str(kmeans_fraction_threshold).replace('.',',')))

    detector.detect({'n_clusters': kmeans_n_clusters, 'fraction_threshold': kmeans_fraction_threshold})

    detector.show_results("true")
    print("done")

    print('cmeans')

    detector = DETECTORS["cmeans"](dataset,
                                   str(file.replace('.csv','')) + '_cmeans_n_clusters_'
                                   + str(str(cmeans_n_clusters).replace('.',',')) +
                                   '_fraction_threshold_' +
                                   str(str(cmeans_fraction_threshold).replace('.',',')))

    detector.detect({'n_clusters': cmeans_n_clusters, 'fraction_threshold': cmeans_fraction_threshold})

    detector.show_results("true")

    print("done")

    print('Fuzzy Hierarchical Clustering')
    # Compute the fuzzy clusters for the test data
    FuzzyOutlierDetection(X_train, X_test, y_test, hk_threshold_distance, hk_threshold_cluster, str(file.replace('.csv','')))

    print("done")


