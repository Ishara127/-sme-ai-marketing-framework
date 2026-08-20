import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   MODEL 3 ENHANCED — ROI CLASSIFIER WITH XGBOOST")
print("   Student: A.S.D. Ishara Lakmali | ID: 15018")
print("=" * 60)

# STEP 1: Load data
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('marketing_campaign_dataset.csv', nrows=10000)
print(f"  Rows: {len(df):,}")

# STEP 2: Clean data
print("\n[STEP 2] Cleaning data...")
df['Acquisition_Cost'] = (df['Acquisition_Cost'].astype(str)
                          .str.replace('$','',regex=False)
                          .str.replace(',','',regex=False))
df['Acquisition_Cost'] = pd.to_numeric(df['Acquisition_Cost'], errors='coerce').fillna(0)

# STEP 3: Feature engineering
print("\n[STEP 3] Feature engineering...")
le = LabelEncoder()
cat_cols = ['Campaign_Type','Target_Audience','Channel_Used',
            'Location','Language','Customer_Segment']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df['CTR']            = df['Clicks'] / (df['Impressions'] + 1)
df['Cost_Per_Click'] = df['Acquisition_Cost'] / (df['Clicks'] + 1)
df['Revenue_Score']  = df['Conversion_Rate'] * df['Engagement_Score']

# STEP 4: Create target
print("\n[STEP 4] Creating ROI categories...")
df['ROI_Category'] = pd.cut(df['ROI'],
                             bins=[0, 3.5, 5.5, 8.1],
                             labels=['Low ROI','Medium ROI','High ROI'])
df = df.dropna(subset=['ROI_Category'])

feature_cols = ['Campaign_Type','Target_Audience','Channel_Used',
                'Conversion_Rate','Acquisition_Cost','Clicks',
                'Impressions','Engagement_Score','Customer_Segment',
                'CTR','Cost_Per_Click','Revenue_Score']

le2 = LabelEncoder()
X   = df[feature_cols]
y   = le2.fit_transform(df['ROI_Category'])
class_names = list(le2.classes_)

print(f"  Classes: {class_names}")
print(f"  Distribution:")
for cls, cnt in zip(class_names, np.bincount(y)):
    print(f"    {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

# STEP 5: Split data
print("\n[STEP 5] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# STEP 6: Train all 3 models
print("\n[STEP 6] Training all 3 models...")

print("  [1/3] Random Forest...")
rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"        Accuracy: {rf_acc*100:.2f}%")

print("  [2/3] Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc  = accuracy_score(y_test, gb_pred)
print(f"        Accuracy: {gb_acc*100:.2f}%")

print("  [3/3] XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    n_jobs=-1, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
print(f"        Accuracy: {xgb_acc*100:.2f}%")

# STEP 7: Results
print("\n" + "=" * 60)
print("   RESULTS COMPARISON")
print("=" * 60)
print(f"\n  Random Forest    : {rf_acc*100:.2f}%")
print(f"  Gradient Boosting: {gb_acc*100:.2f}%")
print(f"  XGBoost          : {xgb_acc*100:.2f}%  ← {'✅ BEST' if xgb_acc == max(rf_acc,gb_acc,xgb_acc) else ''}")

best_acc  = max(rf_acc, gb_acc, xgb_acc)
best_name = ['Random Forest','Gradient Boosting','XGBoost'][[rf_acc,gb_acc,xgb_acc].index(best_acc)]
best_pred = [rf_pred,gb_pred,xgb_pred][[rf_acc,gb_acc,xgb_acc].index(best_acc)]
best_model = [rf,gb,xgb][[rf_acc,gb_acc,xgb_acc].index(best_acc)]

print(f"\n  BEST MODEL : {best_name}")
print(f"  ACCURACY   : {best_acc*100:.2f}%")
print(f"\n  CLASSIFICATION REPORT ({best_name}):")
print(classification_report(y_test, best_pred, target_names=class_names))

# STEP 8: Charts
print("\n[STEP 7] Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model 3 Enhanced — ROI Classifier with XGBoost\n'
             'A.S.D. Ishara Lakmali Gunathilaka | ID: 15018',
             fontsize=13, fontweight='bold')

# Chart 1 — Model comparison
models  = ['Random\nForest','Gradient\nBoosting','XGBoost']
accs    = [rf_acc*100, gb_acc*100, xgb_acc*100]
colors  = ['#3498db','#2ecc71','#e74c3c']
bars    = axes[0,0].bar(models, accs, color=colors, width=0.5)
axes[0,0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_ylim(0, 100)
axes[0,0].axhline(y=33, color='gray', linestyle='--',
                  alpha=0.7, label='Random baseline (33%)')
axes[0,0].legend(fontsize=9)
for bar, acc in zip(bars, accs):
    axes[0,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center',
                   fontweight='bold', fontsize=12)

# Chart 2 — Confusion matrix
cm = confusion_matrix(y_test, best_pred)
im = axes[0,1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0,1].set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
axes[0,1].set_xticks(range(len(class_names)))
axes[0,1].set_yticks(range(len(class_names)))
axes[0,1].set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
axes[0,1].set_yticklabels(class_names, fontsize=9)
axes[0,1].set_ylabel('Actual')
axes[0,1].set_xlabel('Predicted')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[0,1].text(j, i, str(cm[i,j]), ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       color='white' if cm[i,j] > cm.max()/2 else 'black')

# Chart 3 — Feature importance
feat_imp = pd.DataFrame({
    'Feature'   : feature_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=True)
axes[1,0].barh(feat_imp['Feature'], feat_imp['Importance'], color='#3498db')
axes[1,0].set_title(f'Feature Importance\n({best_name})', fontweight='bold')
axes[1,0].set_xlabel('Importance Score')

# Chart 4 — Accuracy improvement
improvement = [(a - 33) for a in accs]
axes[1,1].bar(models, improvement, color=colors, width=0.5)
axes[1,1].set_title('Accuracy Above Random Baseline\n(33% = random guess)',
                    fontweight='bold')
axes[1,1].set_ylabel('Improvement above baseline (%)')
for i, (bar, imp) in enumerate(zip(axes[1,1].patches, improvement)):
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.3,
                   f'+{imp:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('xgboost_comparison.png', dpi=150, bbox_inches='tight')
print("  Chart saved: xgboost_comparison.png")

print("\n" + "=" * 60)
print("   XGBOOST MODEL COMPLETE!")
print(f"   Best Model    : {best_name}")
print(f"   Best Accuracy : {best_acc*100:.2f}%")
print(f"   Improvement   : +{(best_acc-0.33)*100:.1f}% above random baseline")
print("=" * 60)