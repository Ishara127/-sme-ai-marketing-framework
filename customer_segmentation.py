import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   CUSTOMER SEGMENTATION MODEL - K-MEANS CLUSTERING")
print("   Student: A.S.D. Ishara Lakmali | ID: 15018")
print("=" * 60)

# STEP 1: Load data
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('marketing_campaign_dataset.csv')
df_sample = df[:10000].copy()
print(f"  Rows loaded: {len(df_sample):,}")

# STEP 2: Clean Acquisition_Cost
print("\n[STEP 2] Cleaning data...")
df_sample['Acquisition_Cost'] = df_sample['Acquisition_Cost'].astype(str)
df_sample['Acquisition_Cost'] = df_sample['Acquisition_Cost'].str.replace('$', '', regex=False)
df_sample['Acquisition_Cost'] = df_sample['Acquisition_Cost'].str.replace(',', '', regex=False)
df_sample['Acquisition_Cost'] = pd.to_numeric(df_sample['Acquisition_Cost'], errors='coerce')
df_sample['Acquisition_Cost'] = df_sample['Acquisition_Cost'].fillna(0)

# Encode categorical
le = LabelEncoder()
cat_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
            'Location', 'Language', 'Customer_Segment']
for col in cat_cols:
    df_sample[col] = le.fit_transform(df_sample[col].astype(str))
print("  Data cleaned!")

# STEP 3: Select features for clustering
print("\n[STEP 3] Selecting features...")
cluster_features = ['Conversion_Rate', 'Acquisition_Cost', 'Clicks',
                    'Impressions', 'Engagement_Score', 'ROI']
X = df_sample[cluster_features].copy()

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Features: {cluster_features}")
print(f"  Data scaled successfully!")

# STEP 4: Find best K using Elbow Method
print("\n[STEP 4] Finding best number of clusters (Elbow Method)...")
inertias = []
silhouettes = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_)
    silhouettes.append(sil)
    print(f"  K={k} | Inertia: {kmeans.inertia_:.2f} | Silhouette: {sil:.4f}")

best_k = K_range[np.argmax(silhouettes)]
print(f"\n  BEST K = {best_k} (highest silhouette score)")

# STEP 5: Train final model
print(f"\n[STEP 5] Training K-Means with K={best_k}...")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)
df_sample['Cluster'] = kmeans_final.labels_
print(f"  Model trained! {best_k} customer segments identified.")

# STEP 6: Analyze segments
print("\n[STEP 6] Segment Analysis:")
print("-" * 60)
segment_names = {0: "High-Value", 1: "Occasional", 2: "New Prospects",
                 3: "At-Risk", 4: "Loyal", 5: "Dormant", 6: "Premium"}
for cluster in range(best_k):
    cluster_data = df_sample[df_sample['Cluster'] == cluster]
    name = segment_names.get(cluster, f"Segment {cluster}")
    print(f"\n  SEGMENT {cluster} — {name}:")
    print(f"    Count         : {len(cluster_data):,} customers ({len(cluster_data)/len(df_sample)*100:.1f}%)")
    print(f"    Avg ROI       : {cluster_data['ROI'].mean():.2f}")
    print(f"    Avg Clicks    : {cluster_data['Clicks'].mean():.0f}")
    print(f"    Avg Conversion: {cluster_data['Conversion_Rate'].mean():.4f}")
    print(f"    Avg Engagement: {cluster_data['Engagement_Score'].mean():.2f}")
    print(f"    Avg Cost      : ${cluster_data['Acquisition_Cost'].mean():.2f}")

# STEP 7: Charts
print("\n[STEP 7] Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Segmentation — K-Means Clustering\nA.S.D. Ishara Lakmali | ID: 15018',
             fontsize=14, fontweight='bold')

colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#34495e','#f1c40f','#16a085']

# Chart 1 - Elbow Method
axes[0,0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0,0].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0,0].set_title('Elbow Method\n(Finding Best K)', fontweight='bold')
axes[0,0].set_xlabel('Number of Clusters (K)')
axes[0,0].set_ylabel('Inertia')
axes[0,0].legend()

# Chart 2 - Silhouette Scores
axes[0,1].plot(list(K_range), silhouettes, 'go-', linewidth=2, markersize=8)
axes[0,1].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0,1].set_title('Silhouette Score\n(Higher = Better)', fontweight='bold')
axes[0,1].set_xlabel('Number of Clusters (K)')
axes[0,1].set_ylabel('Silhouette Score')
axes[0,1].legend()

# Chart 3 - Cluster Distribution
cluster_counts = df_sample['Cluster'].value_counts().sort_index()
labels = [f"Seg {i}\n{segment_names.get(i, '')}" for i in cluster_counts.index]
axes[1,0].bar(labels, cluster_counts.values,
              color=colors[:best_k], edgecolor='white')
axes[1,0].set_title('Customer Segment Distribution', fontweight='bold')
axes[1,0].set_ylabel('Number of Customers')
for i, (label, val) in enumerate(zip(labels, cluster_counts.values)):
    axes[1,0].text(i, val + 10, str(val), ha='center', fontweight='bold', fontsize=9)

# Chart 4 - PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
for cluster in range(best_k):
    mask = df_sample['Cluster'] == cluster
    name = segment_names.get(cluster, f"Seg {cluster}")
    axes[1,1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=colors[cluster], label=name, alpha=0.5, s=10)
axes[1,1].set_title('Customer Segments Visualization\n(PCA 2D)', fontweight='bold')
axes[1,1].set_xlabel('PCA Component 1')
axes[1,1].set_ylabel('PCA Component 2')
axes[1,1].legend(fontsize=8, markerscale=2)

plt.tight_layout()
plt.savefig('customer_segmentation_results.png', dpi=150, bbox_inches='tight')
print("  Chart saved: customer_segmentation_results.png")

# Final summary
print("\n" + "=" * 60)
print("   CUSTOMER SEGMENTATION COMPLETE!")
print(f"   Total Segments Found : {best_k}")
print(f"   Best Silhouette Score: {max(silhouettes):.4f}")
print(f"   Chart saved to       : customer_segmentation_results.png")
print("=" * 60)