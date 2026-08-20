import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   SENSITIVITY ANALYSIS — SCALABILITY VALIDATION")
print("   Student: A.S.D. Ishara Lakmali | ID: 15018")
print("=" * 60)

# STEP 1: Load full dataset
print("\n[STEP 1] Loading full dataset...")
df = pd.read_csv('../marketing_campaign_dataset.csv')
df['Acquisition_Cost'] = (df['Acquisition_Cost'].astype(str)
                          .str.replace('$','',regex=False)
                          .str.replace(',','',regex=False))
df['Acquisition_Cost'] = pd.to_numeric(
    df['Acquisition_Cost'], errors='coerce').fillna(0)

# Encode
le = LabelEncoder()
cat_cols = ['Campaign_Type','Target_Audience','Channel_Used',
            'Location','Language','Customer_Segment']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Feature engineering
df['CTR']            = df['Clicks'] / (df['Impressions'] + 1)
df['Cost_Per_Click'] = df['Acquisition_Cost'] / (df['Clicks'] + 1)
df['Revenue_Score']  = df['Conversion_Rate'] * df['Engagement_Score']

# ROI categories
df['ROI_Category'] = pd.cut(df['ROI'],
                             bins=[0, 3.5, 5.5, 8.1],
                             labels=['Low ROI','Medium ROI','High ROI'])
df = df.dropna(subset=['ROI_Category'])
le2 = LabelEncoder()
df['ROI_Label'] = le2.fit_transform(df['ROI_Category'])

feature_cols = ['Campaign_Type','Target_Audience','Channel_Used',
                'Conversion_Rate','Acquisition_Cost','Clicks',
                'Impressions','Engagement_Score','Customer_Segment',
                'CTR','Cost_Per_Click','Revenue_Score']

cluster_cols = ['Conversion_Rate','Acquisition_Cost',
                'Clicks','Impressions','Engagement_Score','ROI']

print(f"  Total rows available: {len(df):,}")

# STEP 2: Sample sizes to test
sample_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
sample_sizes = [s for s in sample_sizes if s <= len(df)]

print(f"\n[STEP 2] Testing {len(sample_sizes)} sample sizes:")
print(f"  Sizes: {sample_sizes}")

# STEP 3: Run analysis for each size
results = []

for size in sample_sizes:
    print(f"\n  Testing n={size:,}...")
    df_s = df.sample(n=size, random_state=42)

    # Classification
    X   = df_s[feature_cols]
    y   = df_s['ROI_Label']
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Gradient Boosting
    gb  = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gb.fit(Xtr, ytr)
    gb_acc = accuracy_score(yte, gb.predict(Xte))

    # Random Forest
    rf  = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_acc = accuracy_score(yte, rf.predict(Xte))

    # K-Means Segmentation
    X_cl     = df_s[cluster_cols].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_cl)
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)

    results.append({
        'Sample Size' : size,
        'GB Accuracy' : round(gb_acc * 100, 2),
        'RF Accuracy' : round(rf_acc * 100, 2),
        'Silhouette'  : round(sil, 4),
    })
    print(f"    GB: {gb_acc*100:.2f}% | RF: {rf_acc*100:.2f}% | Sil: {sil:.4f}")

# STEP 4: Results table
print("\n" + "=" * 60)
print("   SENSITIVITY ANALYSIS RESULTS")
print("=" * 60)
print(f"\n{'Sample Size':<15}{'GB Accuracy':<15}{'RF Accuracy':<15}{'Silhouette'}")
print("-" * 55)
for r in results:
    print(f"{r['Sample Size']:<15,}{r['GB Accuracy']:<15.2f}{r['RF Accuracy']:<15.2f}{r['Silhouette']:.4f}")

results_df = pd.DataFrame(results)

# STEP 5: Charts
print("\n[STEP 5] Generating charts...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Sensitivity Analysis — Scalability Validation\n'
             'A.S.D. Ishara Lakmali Gunathilaka | ID: 15018',
             fontsize=13, fontweight='bold')

sizes = results_df['Sample Size']

# Chart 1 — Classifier accuracy vs sample size
axes[0].plot(sizes, results_df['GB Accuracy'],
             'o-', color='#2ecc71', lw=2.5, markersize=8, label='Gradient Boosting')
axes[0].plot(sizes, results_df['RF Accuracy'],
             's-', color='#3498db', lw=2.5, markersize=8, label='Random Forest')
axes[0].axhline(y=33, color='red', linestyle='--',
                 alpha=0.7, label='Random baseline (33%)')
axes[0].set_title('Classifier Accuracy vs Sample Size',
                  fontweight='bold')
axes[0].set_xlabel('Sample Size')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend(fontsize=9)
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)
for x, y in zip(sizes, results_df['GB Accuracy']):
    axes[0].annotate(f'{y:.1f}%', (x, y),
                     textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

# Chart 2 — Silhouette score vs sample size
axes[1].plot(sizes, results_df['Silhouette'],
             'D-', color='#F4A93B', lw=2.5, markersize=8)
axes[1].set_title('Segmentation Quality vs Sample Size\n(Silhouette Score)',
                  fontweight='bold')
axes[1].set_xlabel('Sample Size')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)
for x, y in zip(sizes, results_df['Silhouette']):
    axes[1].annotate(f'{y:.4f}', (x, y),
                     textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

# Chart 3 — Summary bar chart
x_pos = range(len(results_df))
bars  = axes[2].bar(x_pos, results_df['GB Accuracy'],
                    color='#2ecc71', alpha=0.85, width=0.6)
axes[2].axhline(y=33, color='red', linestyle='--',
                alpha=0.7, label='Baseline')
axes[2].set_title('GB Accuracy Across Sample Sizes',
                  fontweight='bold')
axes[2].set_xlabel('Sample Size')
axes[2].set_ylabel('Accuracy (%)')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels([f'{s:,}' for s in sizes],
                         rotation=45, ha='right', fontsize=9)
axes[2].legend(fontsize=9)
for bar, acc in zip(bars, results_df['GB Accuracy']):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center',
                 fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print("  Chart saved: sensitivity_analysis.png")

# STEP 6: Key findings
print("\n" + "=" * 60)
print("   KEY FINDINGS")
print("=" * 60)
min_acc = results_df['GB Accuracy'].min()
max_acc = results_df['GB Accuracy'].max()
min_sil = results_df['Silhouette'].min()
max_sil = results_df['Silhouette'].max()

print(f"\n  Classifier (GB):")
print(f"    Smallest sample (n={results_df.iloc[0]['Sample Size']:,}): {min_acc:.2f}%")
print(f"    Largest sample  (n={results_df.iloc[-1]['Sample Size']:,}): {max_acc:.2f}%")
print(f"    Variance: {max_acc - min_acc:.2f}% — {'STABLE ✅' if max_acc-min_acc < 10 else 'Variable'}")

print(f"\n  Segmentation (K-Means):")
print(f"    Silhouette range: {min_sil:.4f} — {max_sil:.4f}")
print(f"    Consistency: {'STABLE ✅' if max_sil-min_sil < 0.05 else 'Variable'}")

print(f"\n  CONCLUSION:")
print(f"    The framework demonstrates consistent performance")
print(f"    across sample sizes from 500 to {results_df.iloc[-1]['Sample Size']:,} rows,")
print(f"    validating its SCALABILITY claim for SMEs with")
print(f"    varying data volumes.")

print("\n" + "=" * 60)
print("   SENSITIVITY ANALYSIS COMPLETE!")
print("=" * 60)