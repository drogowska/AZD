
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns

def lof():
    df = pd.read_csv("./zad1/data/emotions.csv")

    clf = LocalOutlierFactor(n_neighbors=10)
    df0 = df.drop('label', axis=1)

    y_pred = clf.fit_predict(df0)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df.drop('label', axis=1))
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['label'] = y_pred
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
    plt.show()