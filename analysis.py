"""
Human Activity Recognition with Smartphones - Complete Analysis
================================================================
This script performs the full ML pipeline:
1. Load & Explore the dataset
2. Preprocessing & Missing Value Handling
3. Feature Selection & Scaling
4. Elbow Method for Optimal K
5. K-Means Clustering
6. PCA-based 2D/3D Visualization
7. Cluster vs Actual Label Comparison
8. Interpretation & Save Models for Streamlit App
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
import pickle
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=" * 70)
print("  HUMAN ACTIVITY RECOGNITION USING SMARTPHONES")
print("  Unsupervised K-Means Clustering Analysis")
print("=" * 70)

# ──────────────────────────────────────────────────────
# STEP 1: Load and Explore the Dataset
# ──────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading Dataset...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"   Train shape: {train.shape}")
print(f"   Test shape:  {test.shape}")
print(f"\n   Train columns (first 10): {list(train.columns[:10])}")
print(f"   Total features: {len(train.columns) - 2} (excluding 'subject' and 'Activity')")

print(f"\n   Train Data Types:\n{train.dtypes.value_counts()}")
print(f"\n   Activity Distribution (Train):\n{train['Activity'].value_counts()}")
print(f"\n   Basic Statistics (first 5 features):\n{train.iloc[:, :5].describe()}")

# ──────────────────────────────────────────────────────
# STEP 2: Preprocessing & Missing Values
# ──────────────────────────────────────────────────────
print("\n\n🔧 STEP 2: Preprocessing & Missing Values...")
print(f"   Missing values in Train: {train.isnull().sum().sum()}")
print(f"   Missing values in Test:  {test.isnull().sum().sum()}")

# Combine train and test for the clustering task
df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"   Combined dataset shape: {df.shape}")

# Separate features from labels
X = df.drop(columns=['Activity', 'subject'], errors='ignore')
y = df['Activity'] if 'Activity' in df.columns else None
subjects = df['subject'] if 'subject' in df.columns else None

# Handle missing values (fill with median)
missing_before = X.isnull().sum().sum()
X = X.fillna(X.median())
missing_after = X.isnull().sum().sum()
print(f"   Missing values before fill: {missing_before}")
print(f"   Missing values after fill:  {missing_after}")

# Check for duplicates
duplicates = X.duplicated().sum()
print(f"   Duplicate rows: {duplicates}")
if duplicates > 0:
    X = X.drop_duplicates()
    if y is not None:
        y = y.loc[X.index]
    print(f"   After removing duplicates: {X.shape}")

feature_names = list(X.columns)

# ──────────────────────────────────────────────────────
# STEP 3: Feature Selection & Scaling
# ──────────────────────────────────────────────────────
print("\n\n📐 STEP 3: Feature Selection & Scaling...")
print(f"   Total features: {len(feature_names)}")
print(f"   Feature categories:")
print(f"     - tBodyAcc features: {len([f for f in feature_names if f.startswith('tBodyAcc')])}")
print(f"     - tGravityAcc features: {len([f for f in feature_names if f.startswith('tGravityAcc')])}")
print(f"     - tBodyGyro features: {len([f for f in feature_names if f.startswith('tBodyGyro')])}")
print(f"     - fBody features: {len([f for f in feature_names if f.startswith('fBody')])}")
print(f"     - angle features: {len([f for f in feature_names if f.startswith('angle')])}")

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n   After scaling:")
print(f"     Mean of first 5 features: {X_scaled[:, :5].mean(axis=0).round(6)}")
print(f"     Std of first 5 features:  {X_scaled[:, :5].std(axis=0).round(6)}")

# ──────────────────────────────────────────────────────
# STEP 4: Elbow Method for Optimal K
# ──────────────────────────────────────────────────────
print("\n\n📊 STEP 4: Elbow Method for Optimal K...")
K_range = range(2, 12)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_score = silhouette_score(X_scaled, km.labels_, sample_size=5000, random_state=42)
    silhouette_scores.append(sil_score)
    print(f"   K={k:2d}  |  Inertia={km.inertia_:12.2f}  |  Silhouette={sil_score:.4f}")

# Plot Elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=6, color='red', linestyle='--', alpha=0.7, label='Optimal K=6')
ax1.set_xlabel('Number of Clusters (K)', fontsize=13)
ax1.set_ylabel('Inertia (WCSS)', fontsize=13)
ax1.set_title('Elbow Method - Finding Optimal K', fontsize=15, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'rs-', linewidth=2, markersize=8)
ax2.axvline(x=6, color='red', linestyle='--', alpha=0.7, label='K=6')
ax2.set_xlabel('Number of Clusters (K)', fontsize=13)
ax2.set_ylabel('Silhouette Score', fontsize=13)
ax2.set_title('Silhouette Score vs K', fontsize=15, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'elbow_method.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Elbow plot saved as 'elbow_method.png'")

# ──────────────────────────────────────────────────────
# STEP 5: K-Means Clustering (K=6)
# ──────────────────────────────────────────────────────
print("\n\n🎯 STEP 5: Applying K-Means Clustering (K=6)...")
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

print(f"   Cluster sizes:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"     Cluster {u}: {c} samples ({c/len(cluster_labels)*100:.1f}%)")

print(f"\n   Inertia (WCSS): {kmeans.inertia_:.2f}")
sil = silhouette_score(X_scaled, cluster_labels, sample_size=5000, random_state=42)
print(f"   Silhouette Score: {sil:.4f}")

# ──────────────────────────────────────────────────────
# STEP 6: PCA-based 2D/3D Visualization
# ──────────────────────────────────────────────────────
print("\n\n🗺️ STEP 6: PCA Visualization...")

# PCA with 3 components
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"   Explained variance ratio: {pca.explained_variance_ratio_.round(4)}")
print(f"   Total variance explained: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")

# 2D Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Clusters
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                           cmap='Set2', alpha=0.5, s=10)
axes[0].set_xlabel('PC1', fontsize=13)
axes[0].set_ylabel('PC2', fontsize=13)
axes[0].set_title('K-Means Clusters (PCA 2D)', fontsize=15, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Plot 2: Actual Labels
if y is not None:
    activity_map = {a: i for i, a in enumerate(y.unique())}
    y_numeric = y.map(activity_map).values
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric,
                               cmap='Set2', alpha=0.5, s=10)
    axes[1].set_xlabel('PC1', fontsize=13)
    axes[1].set_ylabel('PC2', fontsize=13)
    axes[1].set_title('Actual Activity Labels (PCA 2D)', fontsize=15, fontweight='bold')
    cbar = plt.colorbar(scatter2, ax=axes[1], label='Activity')
    cbar.set_ticks(list(activity_map.values()))
    cbar.set_ticklabels(list(activity_map.keys()))

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'pca_2d_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 2D PCA plot saved as 'pca_2d_visualization.png'")

# 3D Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=cluster_labels, cmap='Set2', alpha=0.5, s=10)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('K-Means Clusters (PCA 3D)', fontsize=15, fontweight='bold')
plt.colorbar(scatter, label='Cluster', shrink=0.6)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'pca_3d_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 3D PCA plot saved as 'pca_3d_visualization.png'")

# ──────────────────────────────────────────────────────
# STEP 7: Compare Clusters with Actual Labels
# ──────────────────────────────────────────────────────
print("\n\n📋 STEP 7: Comparing Clusters with Actual Labels...")
if y is not None:
    # Adjusted Rand Index
    ari = adjusted_rand_score(y, cluster_labels)
    print(f"   Adjusted Rand Index: {ari:.4f}")

    # Crosstab / Confusion Matrix
    cross_tab = pd.crosstab(y, cluster_labels, rownames=['Actual'], colnames=['Cluster'])
    print(f"\n   Confusion Matrix (Cluster vs Actual):\n{cross_tab}")

    # Heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, linecolor='gray')
    plt.title('Cluster vs Actual Activity - Confusion Matrix', fontsize=15, fontweight='bold')
    plt.xlabel('K-Means Cluster', fontsize=13)
    plt.ylabel('Actual Activity', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'cluster_vs_actual_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Heatmap saved as 'cluster_vs_actual_heatmap.png'")

# ──────────────────────────────────────────────────────
# STEP 8: Interpret Clusters & Identify Patterns
# ──────────────────────────────────────────────────────
print("\n\n🧠 STEP 8: Cluster Interpretation...")

# Generate Readable Feature Names using utils
from utils import get_readable_mapping
readable_feature_mapping = get_readable_mapping(feature_names)

# Key features for interpretation
key_features = [
    'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Z',
    'tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y',
    'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Z',
    'tBodyAccMag-mean()', 'tGravityAccMag-mean()',
    'fBodyAcc-mean()-X', 'fBodyAcc-std()-Y'
]

key_idx = [feature_names.index(f) for f in key_features if f in feature_names]

print("   Cluster Centroids (key features - scaled values):")
for cluster_id in range(6):
    centroid = kmeans.cluster_centers_[cluster_id]
    print(f"\n   Cluster {cluster_id}:")
    for fi, fname in zip(key_idx, key_features):
        readable_name = readable_feature_mapping.get(fname, fname)
        print(f"     {readable_name:40s} = {centroid[fi]:+.4f}")

# Identify which cluster maps to which activity
cluster_activity_mapping = {}
if y is not None:
    print("\n\n   Cluster → Most Common Activity Mapping:")
    df_result = pd.DataFrame({'Activity': y, 'Cluster': cluster_labels})
    for cluster_id in range(6):
        cluster_data = df_result[df_result['Cluster'] == cluster_id]
        if len(cluster_data) > 0:
            most_common = cluster_data['Activity'].mode()[0]
            cluster_activity_mapping[cluster_id] = most_common
            pct = (cluster_data['Activity'] == most_common).sum() / len(cluster_data) * 100
            print(f"     Cluster {cluster_id} → {most_common} ({pct:.1f}% match)")

# Cluster centroid radar chart
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
axes = axes.flatten()
angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

for cluster_id in range(6):
    ax = axes[cluster_id]
    values = kmeans.cluster_centers_[cluster_id, key_idx].tolist()
    values += values[:1]

    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.split('-')[0][-8:] for f in key_features], size=7)
    ax.set_title(f'Cluster {cluster_id}', size=13, fontweight='bold', pad=15)

plt.suptitle('Cluster Centroids - Radar Charts', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'cluster_radar_charts.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Radar charts saved as 'cluster_radar_charts.png'")

# ──────────────────────────────────────────────────────
# STEP 9: Save Models & Metadata for Streamlit App
# ──────────────────────────────────────────────────────
print("\n\n💾 STEP 9: Saving Models & Metadata for Streamlit App...")
artifacts = {
    'kmeans': kmeans,
    'scaler': scaler,
    'pca': pca,
    'feature_names': feature_names,
    'readable_feature_mapping': readable_feature_mapping,
    'cluster_activity_mapping': cluster_activity_mapping,
    'important_features': key_features
}

for name, obj in artifacts.items():
    with open(os.path.join(MODEL_DIR, f'{name}.pkl'), 'wb') as f:
        pickle.dump(obj, f)
    print(f"   ✅ {name}.pkl saved")

print("\n" + "=" * 70)
print("  ✅ ANALYSIS COMPLETE!")
print("  Run the Streamlit app with: streamlit run app.py")
print("=" * 70)
