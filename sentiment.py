import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from tqdm import tqdm

# --- CONFIG ---
CSV_PATH = 'C://Users//shah_kr//Desktop//Projects//VirtualAgent//Sentiment//movies.csv'
NUM_POINTS = 50
CHUNK_SIZE = 5
N_CLUSTERS = 6
SAMPLES_PER_CLUSTER = 5

# --- Sentiment analyzer ---
analyzer = SentimentIntensityAnalyzer()

def get_normalized_arc(plot, chunk_size=CHUNK_SIZE, num_points=NUM_POINTS):
    sentences = re.split(r'(?<=[.!?])\s+', str(plot).strip())
    if len(sentences) < 2:
        return None
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    scores = [analyzer.polarity_scores(chunk)['compound'] for chunk in chunks]
    if len(scores) < 2 or len(set(scores)) == 1:
        return None
    x_old = np.linspace(0, 1, num=len(scores))
    try:
        f = interp1d(x_old, scores, kind='linear', fill_value="extrapolate", bounds_error=False)
        x_new = np.linspace(0, 1, num=num_points)
        return f(x_new).tolist()
    except:
        return None

# --- Load and process data ---
print("Loading and processing data...")
from tqdm import tqdm
tqdm.pandas()  
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['Plot'])
df = df[df['Plot'].str.len() > 100]
df['Normalized_Arc'] = df['Plot'].progress_apply(get_normalized_arc)
df = df.dropna(subset=['Normalized_Arc'])

# --- Smoothing arcs ---
print("Smoothing arcs...")
X = np.array([gaussian_filter1d(arc, sigma=2) for arc in df['Normalized_Arc']])

# --- Clustering ---
print("Clustering arcs...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- Plot cluster centroids ---
print("Plotting cluster centroids...")
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, N_CLUSTERS))

for i, center in enumerate(kmeans.cluster_centers_):
    plt.plot(center, label=f"Cluster {i+1}", linewidth=2)

plt.title("Emotional Arc Centroids by Cluster", fontsize=16)
plt.xlabel("Story Progression")
plt.ylabel("Emotional Valence (Fortunes)")
plt.axhline(0, linestyle='--', color='gray')
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()

# --- t-SNE projection ---
print("Running t-SNE for 2D visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# --- t-SNE Plot ---
plt.figure(figsize=(10, 6))
for i in range(N_CLUSTERS):
    plt.scatter(X_tsne[df['Cluster'] == i, 0], X_tsne[df['Cluster'] == i, 1], alpha=0.6, label=f"Cluster {i+1}")
plt.title("t-SNE of Emotional Arcs (by Cluster)", fontsize=16)
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# --- Plot 5 sample arcs per cluster ---
print("Plotting 5 sample arcs per cluster...")
for cluster_id in range(N_CLUSTERS):
    cluster_df = df[df['Cluster'] == cluster_id].sample(n=min(SAMPLES_PER_CLUSTER, len(df[df['Cluster'] == cluster_id])), random_state=1)
    plt.figure(figsize=(10, 5))
    for _, row in cluster_df.iterrows():
        arc = gaussian_filter1d(row['Normalized_Arc'], sigma=2)
        plt.plot(arc, alpha=0.8, label=row['Title'][:30])
    plt.title(f"Cluster {cluster_id + 1} – Sample Emotional Arcs", fontsize=14)
    plt.xlabel("Story Progression")
    plt.ylabel("Emotional Valence")
    plt.axhline(0, linestyle='--', color='gray')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
