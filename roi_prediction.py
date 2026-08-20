import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   ROI PREDICTION MODEL - AI MARKETING FRAMEWORK")
print("   Student: A.S.D. Ishara Lakmali | ID: 15018")
print("=" * 60)

# STEP 1: Load data
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('marketing_campaign_dataset.csv')
print(f"  Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# STEP 2: Encode categorical columns
print("\n[STEP 2] Encoding categorical columns...")
le = LabelEncoder()
cat_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
            'Location', 'Language', 'Customer_Segment']
df_model = df.copy()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))
# Fix dollar sign and comma in Acquisition_Cost
print("  Cleaning Acquisition_Cost column...")
df_model['Acquisition_Cost'] = df_model['Acquisition_Cost'].astype(str)
df_model['Acquisition_Cost'] = df_model['Acquisition_Cost'].str.replace('$', '', regex=False)
df_model['Acquisition_Cost'] = df_model['Acquisition_Cost'].str.replace(',', '', regex=False)
df_model['Acquisition_Cost'] = pd.to_numeric(df_model['Acquisition_Cost'], errors='coerce')
df_model['Acquisition_Cost'] = df_model['Acquisition_Cost'].fillna(0)
print("  Acquisition_Cost cleaned!")

# STEP 3: Features and target
print("\n[STEP 3] Preparing features...")
# New engineered features
df_model['CTR'] = df_model['Clicks'] / (df_model['Impressions'] + 1)
df_model['Cost_Per_Click'] = df_model['Acquisition_Cost'] / (df_model['Clicks'] + 1)
df_model['Revenue_Score'] = df_model['Conversion_Rate'] * df_model['Engagement_Score']

feature_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
                'Conversion_Rate', 'Acquisition_Cost', 'Clicks',
                'Impressions', 'Engagement_Score', 'Customer_Segment',
                'CTR', 'Cost_Per_Click', 'Revenue_Score']

X = df_model[feature_cols]
y = df_model['ROI']

# Use sample for speed
X = X[:10000]
y = y[:10000]
print(f"  Features: {len(feature_cols)} | Rows: {len(X):,}")

# STEP 4: Split data
print("\n[STEP 4] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# STEP 5: Train models
print("\n[STEP 5] Training models...")
print("  Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2  = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

print("  Training Random Forest (takes ~30 seconds)...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2  = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

# STEP 6: Results
print("\n" + "=" * 60)
print("   RESULTS")
print("=" * 60)
print(f"\n  LINEAR REGRESSION:")
print(f"    R² Score : {lr_r2:.4f} ({lr_r2*100:.1f}%)")
print(f"    MAE      : {lr_mae:.4f}")
print(f"    MSE      : {lr_mse:.4f}")

print(f"\n  RANDOM FOREST:")
print(f"    R² Score : {rf_r2:.4f} ({rf_r2*100:.1f}%)")
print(f"    MAE      : {rf_mae:.4f}")
print(f"    MSE      : {rf_mse:.4f}")

best_r2 = max(lr_r2, rf_r2)
winner  = "Random Forest" if rf_r2 >= lr_r2 else "Linear Regression"
best_pred = rf_pred if rf_r2 >= lr_r2 else lr_pred
print(f"\n  BEST MODEL : {winner}")
print(f"  BEST R²    : {best_r2:.4f}")

# STEP 7: Feature importance
print("\n  FEATURE IMPORTANCE (Random Forest):")
feat_df = pd.DataFrame({'Feature': feature_cols,
                        'Importance': rf.feature_importances_}
                       ).sort_values('Importance', ascending=False)
for _, row in feat_df.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"    {row['Feature']:<20} {row['Importance']:.4f}  {bar}")

# STEP 8: Charts
print("\n[STEP 6] Saving charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ROI Prediction Model\nA.S.D. Ishara Lakmali | ID: 15018',
             fontsize=14, fontweight='bold')

# Chart 1 - Model Comparison
bars = axes[0,0].bar(['Linear\nRegression','Random\nForest'],
                     [lr_r2, rf_r2], color=['#3498db','#2ecc71'])
axes[0,0].set_title('Model Comparison — R² Score', fontweight='bold')
axes[0,0].set_ylim(0, 1)
for bar, score in zip(bars, [lr_r2, rf_r2]):
    axes[0,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01,
                   f'{score:.4f}', ha='center', fontweight='bold')

# Chart 2 - Actual vs Predicted
idx = np.random.choice(len(y_test), 300, replace=False)
axes[0,1].scatter(y_test.values[idx], best_pred[idx], alpha=0.4, color='#3498db', s=15)
mn, mx = y_test.min(), y_test.max()
axes[0,1].plot([mn,mx],[mn,mx],'r--', label='Perfect prediction')
axes[0,1].set_title(f'Actual vs Predicted ROI\n({winner})', fontweight='bold')
axes[0,1].set_xlabel('Actual ROI')
axes[0,1].set_ylabel('Predicted ROI')
axes[0,1].legend()

# Chart 3 - Feature Importance
feat_sorted = feat_df.sort_values('Importance', ascending=True)
axes[1,0].barh(feat_sorted['Feature'], feat_sorted['Importance'], color='#3498db')
axes[1,0].set_title('Feature Importance\n(Random Forest)', fontweight='bold')
axes[1,0].set_xlabel('Importance Score')

# Chart 4 - ROI Distribution
axes[1,1].hist(y_test, bins=30, alpha=0.6, color='#3498db', label='Actual ROI')
axes[1,1].hist(best_pred, bins=30, alpha=0.6, color='#e74c3c', label='Predicted ROI')
axes[1,1].set_title('ROI Distribution\nActual vs Predicted', fontweight='bold')
axes[1,1].set_xlabel('ROI Value')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('roi_prediction_results.png', dpi=150, bbox_inches='tight')
print("  Chart saved: roi_prediction_results.png")

print("\n" + "=" * 60)
print("   ROI PREDICTION MODEL COMPLETE!")
print("=" * 60)