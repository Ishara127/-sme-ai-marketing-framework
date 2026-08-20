import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   CAMPAIGN CLASSIFIER MODEL")
print("   Student: A.S.D. Ishara Lakmali | ID: 15018")
print("=" * 60)

# STEP 1: Load data
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('marketing_campaign_dataset.csv')
df = df[:10000].copy()
print(f"  Rows loaded: {len(df):,}")

# STEP 2: Clean data
print("\n[STEP 2] Cleaning data...")
df['Acquisition_Cost'] = df['Acquisition_Cost'].astype(str)
df['Acquisition_Cost'] = df['Acquisition_Cost'].str.replace('$', '', regex=False)
df['Acquisition_Cost'] = df['Acquisition_Cost'].str.replace(',', '', regex=False)
df['Acquisition_Cost'] = pd.to_numeric(df['Acquisition_Cost'], errors='coerce').fillna(0)

# STEP 3: Create ROI category target
print("\n[STEP 3] Creating target variable...")
df['ROI_Category'] = pd.cut(df['ROI'],
                             bins=[0, 3.5, 5.5, 8.1],
                             labels=['Low ROI', 'Medium ROI', 'High ROI'])
df = df.dropna(subset=['ROI_Category'])
print(f"  ROI Categories created:")
print(f"  {df['ROI_Category'].value_counts().to_string()}")

# STEP 4: Encode features
print("\n[STEP 4] Encoding features...")
le = LabelEncoder()
cat_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
            'Location', 'Language', 'Customer_Segment']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Engineer features
df['CTR'] = df['Clicks'] / (df['Impressions'] + 1)
df['Cost_Per_Click'] = df['Acquisition_Cost'] / (df['Clicks'] + 1)
df['Revenue_Score'] = df['Conversion_Rate'] * df['Engagement_Score']

feature_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
                'Conversion_Rate', 'Acquisition_Cost', 'Clicks',
                'Impressions', 'Engagement_Score', 'Customer_Segment',
                'CTR', 'Cost_Per_Click', 'Revenue_Score']

X = df[feature_cols]
y = le.fit_transform(df['ROI_Category'])
class_names = ['High ROI', 'Low ROI', 'Medium ROI']

print(f"  Features: {len(feature_cols)}")

# STEP 5: Split data
print("\n[STEP 5] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# STEP 6: Train models
print("\n[STEP 6] Training models...")
print("  Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"  Random Forest Accuracy: {rf_acc*100:.2f}%")

print("  Training Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"  Gradient Boosting Accuracy: {gb_acc*100:.2f}%")

# STEP 7: Results
print("\n" + "=" * 60)
print("   RESULTS")
print("=" * 60)
best_pred = rf_pred if rf_acc >= gb_acc else gb_pred
best_name = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"
best_acc  = max(rf_acc, gb_acc)
best_model = rf if rf_acc >= gb_acc else gb

print(f"\n  BEST MODEL : {best_name}")
print(f"  ACCURACY   : {best_acc*100:.2f}%")
print(f"\n  CLASSIFICATION REPORT:")
print(classification_report(y_test, best_pred, target_names=class_names))

# STEP 8: Sample predictions
print("\n  SAMPLE PREDICTIONS (10 rows):")
print(f"  {'Actual':<15} {'Predicted':<15} {'Correct?'}")
print(f"  {'-'*40}")
for i in range(10):
    actual    = class_names[y_test[i]]
    predicted = class_names[best_pred[i]]
    correct   = "✓" if y_test[i] == best_pred[i] else "✗"
    print(f"  {actual:<15} {predicted:<15} {correct}")

# STEP 9: Charts
print("\n[STEP 7] Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Campaign ROI Classifier Results\nA.S.D. Ishara Lakmali | ID: 15018',
             fontsize=14, fontweight='bold')

# Chart 1 - Model Accuracy Comparison
models  = ['Random\nForest', 'Gradient\nBoosting']
accs    = [rf_acc, gb_acc]
colors  = ['#3498db', '#2ecc71']
bars    = axes[0,0].bar(models, accs, color=colors, width=0.5)
axes[0,0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_ylim(0, 1)
axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
axes[0,0].legend()
for bar, acc in zip(bars, accs):
    axes[0,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01,
                   f'{acc*100:.1f}%', ha='center', fontweight='bold', fontsize=12)

# Chart 2 - Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
im = axes[0,1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0,1].set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
axes[0,1].set_xticks(range(len(class_names)))
axes[0,1].set_yticks(range(len(class_names)))
axes[0,1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[0,1].set_yticklabels(class_names, fontsize=9)
axes[0,1].set_ylabel('Actual')
axes[0,1].set_xlabel('Predicted')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[0,1].text(j, i, str(cm[i,j]), ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       color='white' if cm[i,j] > cm.max()/2 else 'black')

# Chart 3 - Feature Importance
feat_df = pd.DataFrame({'Feature': feature_cols,
                        'Importance': best_model.feature_importances_}
                       ).sort_values('Importance', ascending=True)
axes[1,0].barh(feat_df['Feature'], feat_df['Importance'], color='#3498db')
axes[1,0].set_title('Feature Importance', fontweight='bold')
axes[1,0].set_xlabel('Importance Score')

# Chart 4 - ROI Category Distribution
roi_counts = df['ROI_Category'].value_counts()
axes[1,1].pie(roi_counts.values,
              labels=roi_counts.index,
              colors=['#2ecc71','#f39c12','#e74c3c'],
              autopct='%1.1f%%', startangle=90,
              textprops={'fontsize': 11})
axes[1,1].set_title('ROI Category Distribution\nin Dataset', fontweight='bold')

plt.tight_layout()
plt.savefig('campaign_classifier_results.png', dpi=150, bbox_inches='tight')
print("  Chart saved: campaign_classifier_results.png")

print("\n" + "=" * 60)
print("   CAMPAIGN CLASSIFIER COMPLETE!")
print(f"   Best Model    : {best_name}")
print(f"   Best Accuracy : {best_acc*100:.2f}%")
print(f"   Chart saved   : campaign_classifier_results.png")
print("=" * 60)