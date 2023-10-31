from matplotlib import pyplot as plt
import pandas as pd
from pyod.models.abod import ABOD
from sklearn.manifold import TSNE
import seaborn as sns

def abod(x_test,x_train, y_train, y_test):
    abod = ABOD(method='fast')
    abod.fit(x_train)

    y_train_pred = abod.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = abod.decision_scores_ 

    y_test_pred = abod.predict(x_test)  
    y_test_scores = abod.decision_function(x_test)  
  
    # print(y_test_scores)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(x_test)
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['label'] = y_test_pred
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
    plt.show()
