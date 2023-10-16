import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# wykres rozkładu danych w poszczególnych klasach
df = pd.read_csv("./zad1/data/emotions.csv")
palette = sns.color_palette("dark")
plt.figure(figsize=(8, 6))
bars = df["label"].value_counts().plot(kind='bar', color=palette)
plt.ylabel('Value Counts')
plt.xticks(rotation=0)

for i, v in enumerate(df["label"].value_counts()):
    plt.text(i, v, str(v), ha='center', va='bottom', fontsize=12)

# plt.show()


positive = df.loc[df["label"]=='POSITIVE']
negative = df.loc[df["label"]=='NEGATIVE']
neutral = df.loc[df["label"]=='NEUTRAL']

fig,axes = plt.subplots(nrows=3, ncols=1, dpi=50, figsize=(25,12))

positive.loc[2, 'fft_0_a':'fft_749_a'].plot(title='fft of positive columns', color = 'tab:purple', ax=axes[0], linewidth=0.5)
negative.loc[0, 'fft_0_a':'fft_749_a'].plot(title='fft of negative columns', color = 'tab:blue', ax=axes[1], linewidth=0.5)
neutral.loc[1, 'fft_0_a':'fft_749_a'].plot(title='fft of neutral columns', color = 'tab:green', ax=axes[2], linewidth=0.5)

plt.show()


# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df.drop('label', axis=1))
tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
tsne_df['label'] = df['label']
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
plt.show()