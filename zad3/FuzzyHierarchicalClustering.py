import numpy as np

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt



def fuzzy_hierarchical_clustering(data, threshold_cluster):
    # Calculate the pairwise distances between data points
    dist = cdist(data, data)

    # Convert the distances to similarities using a Gaussian kernel
    sigma = np.mean(dist)
    s = np.exp(-(dist / sigma) ** 2)

    # Perform hierarchical clustering
    linkage_matrix = linkage(s, method='average')

    # Calculate the fuzzy clusters using the threshold
    f_clusters = fcluster(linkage_matrix, threshold_cluster, criterion='distance')

    return f_clusters


def FuzzyOutlierDetection(X_train, X_test, y_test, threshold, threshold_cluster, file):

    # Normalize the training and test data
    # Train the Fuzzy Hierarchical Clustering model on the training data
    f_clusters_train = fuzzy_hierarchical_clustering(X_train, threshold_cluster)

    # Compute the fuzzy clusters and membership scores for the test data
    f_clusters_test = fuzzy_hierarchical_clustering(X_test, threshold_cluster)
    mu = np.zeros((X_test.shape[0], len(np.unique(f_clusters_train))))
    for i, f_cluster in enumerate(np.unique(f_clusters_train)):
        indices = np.where(f_clusters_test == f_cluster)[0]
        if len(indices) > 0:
            dist = cdist(X_test[indices], X_train[f_clusters_train == f_cluster])
            sigma = np.mean(dist)
            s = np.exp(-(dist / sigma) ** 2)
            for j in range(len(indices)):
                mu[indices[j], i] = np.max(s[j])

    # Compute the outlier scores for the test data
    scores = np.mean(mu, axis=1)

    # Classify the test data as normal or anomaly based on the threshold score
    y_pred = (scores >= threshold).astype(int)
    #print(scores)
    #print(y_pred)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    for label in set(np.unique(y_pred)) - {-1}:
        plt.scatter(X_test[y_pred == label, 0], X_test[y_pred == label, 1])

    plt.title('Fuzzy Hierarchical Clustering for Outlier Detection ' + file)
    plt.show()
