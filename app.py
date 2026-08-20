from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import json, os, uuid
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import csv, io

app = Flask(__name__)
app.secret_key = 'sme_ai_research_15018'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Helper: clean uploaded dataframe ──────────────────────
def clean_df(df):
    # Fix currency columns
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.contains(r'[\$,]').any():
                df[col] = (df[col].astype(str)
                           .str.replace('$','',regex=False)
                           .str.replace(',','',regex=False))
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def encode_df(df):
    le = LabelEncoder()
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        try:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        except:
            pass
    return df_enc

# ── Default dataset (fallback) ─────────────────────────────
def load_default():
    path = '../marketing_campaign_dataset.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return clean_df(df)
    return None

# ── HOME ───────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

# ── UPLOAD ─────────────────────────────────────────────────
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(url_for('upload'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(url_for('upload'))

        if not file.filename.endswith('.csv'):
            flash('Please upload a CSV file only!', 'error')
            return redirect(url_for('upload'))

        # Save file
        fname  = f"{uuid.uuid4().hex}.csv"
        fpath  = os.path.join(UPLOAD_FOLDER, fname)
        file.save(fpath)

        # Read & validate
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            flash(f'Could not read CSV: {str(e)}', 'error')
            return redirect(url_for('upload'))

        if len(df) < 10:
            flash('CSV needs at least 10 rows of data!', 'error')
            return redirect(url_for('upload'))

        session['csv_path']   = fpath
        session['csv_name']   = file.filename
        session['csv_rows']   = len(df)
        session['csv_cols']   = list(df.columns)
        return redirect(url_for('preview'))

    return render_template('upload.html')

# ── PREVIEW ────────────────────────────────────────────────
@app.route('/preview')
def preview():
    if 'csv_path' not in session:
        return redirect(url_for('upload'))

    df      = pd.read_csv(session['csv_path'])
    df      = clean_df(df)
    preview = df.head(5).to_dict('records')
    cols    = list(df.columns)
    dtypes  = {c: str(df[c].dtype) for c in cols}
    missing = df.isnull().sum().to_dict()
    numerics = df.select_dtypes(include=np.number).columns.tolist()

    return render_template('preview.html',
                           preview=preview,
                           cols=cols,
                           dtypes=dtypes,
                           missing=missing,
                           numerics=numerics,
                           filename=session.get('csv_name',''),
                           rows=session.get('csv_rows',0))

# ── ANALYZE ────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'csv_path' not in session:
        return redirect(url_for('upload'))

    df      = pd.read_csv(session['csv_path'])
    df      = clean_df(df)
    df_enc  = encode_df(df)
    numerics = df.select_dtypes(include=np.number).columns.tolist()
    results  = {}

    # ── 1. Basic Stats ──────────────────────────────────────
    stats = {}
    for col in numerics[:8]:
        stats[col] = {
            'mean' : round(float(df[col].mean()), 3),
            'max'  : round(float(df[col].max()),  3),
            'min'  : round(float(df[col].min()),  3),
            'std'  : round(float(df[col].std()),  3),
        }
    results['stats']    = stats
    results['n_rows']   = len(df)
    results['n_cols']   = len(df.columns)
    results['numerics'] = numerics

    # ── 2. Customer Segmentation ───────────────────────────
    seg_cols = [c for c in numerics if c in df_enc.columns][:6]
    if len(seg_cols) >= 2:
        try:
            X        = df_enc[seg_cols].fillna(0)
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n        = min(5, max(2, len(df)//50))
            km       = KMeans(n_clusters=n, random_state=42, n_init=10)
            km.fit(X_scaled)
            df['Cluster'] = km.labels_

            seg_summary = []
            for c in range(n):
                cdata = df[df['Cluster']==c]
                row   = {'id': c, 'count': len(cdata),
                         'pct': round(len(cdata)/len(df)*100, 1)}
                for col in numerics[:4]:
                    row[f'avg_{col}'] = round(float(cdata[col].mean()), 2)
                seg_summary.append(row)

            sil = round(float(silhouette_score(X_scaled, km.labels_)), 4)
            results['segmentation'] = {
                'segments'   : seg_summary,
                'k'          : n,
                'silhouette' : sil,
                'features'   : seg_cols,
                'pie': {
                    'labels': [f"Segment {s['id']}" for s in seg_summary],
                    'values': [s['count'] for s in seg_summary],
                }
            }
        except Exception as e:
            results['seg_error'] = str(e)

    # ── 3. Top Column Analysis ─────────────────────────────
    if len(numerics) >= 2:
        target_col = request.form.get('target_col', numerics[0])
        feature_cols = [c for c in numerics if c != target_col][:8]

        if len(feature_cols) >= 1:
            try:
                X = df_enc[feature_cols].fillna(0)
                y = df_enc[target_col].fillna(0)
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(
                    n_estimators=10, random_state=42, n_jobs=-1)
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                r2   = round(float(r2_score(yte, pred)), 4)

                feat_imp = dict(zip(feature_cols,
                    [round(float(v), 4) for v in model.feature_importances_]))
                feat_sorted = dict(sorted(feat_imp.items(),
                                         key=lambda x: x[1], reverse=True))

                results['prediction'] = {
                    'target'     : target_col,
                    'r2'         : r2,
                    'features'   : feature_cols,
                    'importance' : feat_sorted,
                    'bar': {
                        'labels': list(feat_sorted.keys()),
                        'values': list(feat_sorted.values()),
                    }
                }
            except Exception as e:
                results['pred_error'] = str(e)

    # ── 4. Column distributions ────────────────────────────
    dist_data = {}
    for col in numerics[:4]:
        counts, bins = np.histogram(df[col].dropna(), bins=8)
        dist_data[col] = {
            'labels': [f"{bins[i]:.1f}" for i in range(len(bins)-1)],
            'values': [int(v) for v in counts]
        }
    results['distributions'] = dist_data

    target_col = request.form.get('target_col', numerics[0] if numerics else '')
    return render_template('results.html',
                           results=results,
                           results_json=json.dumps(results),
                           target_col=target_col,
                           filename=session.get('csv_name',''),
                           cols=numerics)

# ── DEFAULT DEMO ───────────────────────────────────────────
@app.route('/demo')
def demo():
    df = load_default()
    if df is None:
        flash('Default dataset not found!', 'error')
        return redirect(url_for('home'))
    stats = {
        'total_campaigns'  : len(df),
        'avg_roi'          : round(float(df['ROI'].mean()), 2),
        'avg_conversion'   : round(float(df['Conversion_Rate'].mean()*100), 1),
        'total_clicks'     : int(df['Clicks'].sum()),
        'total_impressions': int(df['Impressions'].sum()),
        'top_channel'      : str(df.groupby('Channel_Used')['ROI'].mean().idxmax()),
        'campaign_types'   : df['Campaign_Type'].value_counts().to_dict(),
    }
    return render_template('index.html', stats=stats)

@app.route('/demo/roi')
def demo_roi():
    df = load_default()
    if df is None:
        return redirect(url_for('home'))
    roi_by_type    = df.groupby('Campaign_Type')['ROI'].mean().round(2)
    roi_by_channel = df.groupby('Channel_Used')['ROI'].mean().round(2)
    roi_dist       = pd.cut(df['ROI'], bins=6).value_counts().sort_index()
    data = {
        'roi_types'   : {'labels':list(roi_by_type.index), 'values':list(roi_by_type.values)},
        'roi_channels': {'labels':list(roi_by_channel.index), 'values':list(roi_by_channel.values)},
        'roi_dist'    : {'labels':[str(i) for i in roi_dist.index], 'values':[int(v) for v in roi_dist.values]},
        'feature_importance': {
            'labels':['Acquisition Cost','Impressions','Cost Per Click','CTR','Clicks',
                      'Revenue Score','Conversion Rate','Channel','Engagement',
                      'Customer Segment','Target Audience','Campaign Type'],
            'values':[0.1297,0.1276,0.1191,0.1183,0.1135,0.0926,0.0649,0.0521,0.0488,0.0478,0.0461,0.0396]
        }
    }
    return render_template('roi.html', data=json.dumps(data),
                           avg_roi=round(float(df['ROI'].mean()),2),
                           max_roi=round(float(df['ROI'].max()),2),
                           min_roi=round(float(df['ROI'].min()),2))

@app.route('/demo/segments')
def demo_segments():
    segments_data = [
        {'id':0,'name':'High-Value',   'count':1285,'pct':12.9,'avg_roi':5.21,'avg_clicks':612,'avg_cost':11240,'color':'#e74c3c','tip':'Target with premium offers and loyalty rewards.'},
        {'id':1,'name':'Occasional',   'count':1198,'pct':12.0,'avg_roi':4.87,'avg_clicks':498,'avg_cost':12100,'color':'#3498db','tip':'Re-engage with seasonal promotions.'},
        {'id':2,'name':'New Prospects','count':1342,'pct':13.4,'avg_roi':4.12,'avg_clicks':423,'avg_cost':9870, 'color':'#2ecc71','tip':'Nurture with welcome campaigns.'},
        {'id':3,'name':'At-Risk',      'count':1156,'pct':11.6,'avg_roi':3.71,'avg_clicks':433,'avg_cost':14912,'color':'#f39c12','tip':'Win back with personalised discounts.'},
        {'id':4,'name':'Loyal',        'count':1204,'pct':12.0,'avg_roi':5.89,'avg_clicks':701,'avg_cost':10234,'color':'#9b59b6','tip':'Upsell higher-value products.'},
        {'id':5,'name':'Dormant',      'count':1342,'pct':13.4,'avg_roi':6.60,'avg_clicks':382,'avg_cost':14777,'color':'#1abc9c','tip':'Reactivate with bold incentive.'},
        {'id':6,'name':'Premium',      'count':1270,'pct':12.7,'avg_roi':4.24,'avg_clicks':479,'avg_cost':9432, 'color':'#e67e22','tip':'Lowest cost — scale ad spend here.'},
        {'id':7,'name':'Segment 7',    'count':1125,'pct':11.2,'avg_roi':3.45,'avg_clicks':513,'avg_cost':10918,'color':'#34495e','tip':'Analyse further for insights.'},
    ]
    pie_data = {'labels':[s['name'] for s in segments_data],
                'values':[s['count'] for s in segments_data],
                'colors':[s['color'] for s in segments_data]}
    bar_data = {'labels':[s['name'] for s in segments_data],
                'values':[s['avg_roi'] for s in segments_data],
                'colors':[s['color'] for s in segments_data]}
    return render_template('segments.html',
                           segments=segments_data,
                           pie_data=json.dumps(pie_data),
                           bar_data=json.dumps(bar_data),
                           total=sum(s['count'] for s in segments_data))

@app.route('/demo/campaigns')
def demo_campaigns():
    df = load_default()
    if df is None:
        return redirect(url_for('home'))
    ch_roi  = df.groupby('Channel_Used')['ROI'].mean().round(2)
    ch_conv = df.groupby('Channel_Used')['Conversion_Rate'].mean().round(4)
    top5    = df.nlargest(5,'ROI')[['Campaign_Type','Channel_Used','ROI','Clicks','Conversion_Rate']]
    bot5    = df.nsmallest(5,'ROI')[['Campaign_Type','Channel_Used','ROI','Clicks','Conversion_Rate']]
    data = {
        'channels':{'labels':list(ch_roi.index),'roi':list(ch_roi.values),
                    'conversion':[round(v*100,2) for v in ch_conv.values],'clicks':[],'cost':[]},
        'accuracy':{'rf':38.75,'gb':42.25},
        'roi_cats':{'High ROI':42.2,'Medium ROI':32.8,'Low ROI':25.0},
    }
    top5_l = top5.to_dict('records')
    bot5_l = bot5.to_dict('records')
    for r in top5_l+bot5_l:
        r['ROI']             = round(float(r['ROI']),2)
        r['Conversion_Rate'] = round(float(r['Conversion_Rate'])*100,2)
        r['Clicks']          = int(r['Clicks'])
    return render_template('campaigns.html',
                           data=json.dumps(data),
                           top5=top5_l, bot5=bot5_l)
# ── EXPORT CSV ─────────────────────────────────────────────
@app.route('/export/csv')
def export_csv():
    if 'csv_path' not in session:
        return redirect(url_for('home'))

    df = pd.read_csv(session['csv_path'])
    df = clean_df(df)
    df_enc = encode_df(df)
    numerics = df.select_dtypes(include=np.number).columns.tolist()

    # Run segmentation
    export_df = df.copy()
    seg_cols = [c for c in numerics if c in df_enc.columns][:6]
    if len(seg_cols) >= 2:
        try:
            X        = df_enc[seg_cols].fillna(0)
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n        = min(5, max(2, len(df)//50))
            km       = KMeans(n_clusters=n, random_state=42, n_init=10)
            km.fit(X_scaled)
            export_df['AI_Segment'] = km.labels_
        except:
            export_df['AI_Segment'] = 0

    # Add summary stats
    for col in numerics[:4]:
        mean_val = df[col].mean()
        export_df[f'{col}_vs_avg'] = df[col].apply(
            lambda x: 'Above Average' if x > mean_val else 'Below Average')

    # Convert to CSV
    output  = io.StringIO()
    writer  = csv.writer(output)
    writer.writerow(export_df.columns.tolist())
    for _, row in export_df.iterrows():
        writer.writerow(row.tolist())

    output.seek(0)
    filename = session.get('csv_name', 'results').replace('.csv','')

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename={filename}_AI_Analysis.csv'}
    )

# ── EXPORT SUMMARY ──────────────────────────────────────────
@app.route('/export/summary')
def export_summary():
    if 'csv_path' not in session:
        return redirect(url_for('home'))

    df       = pd.read_csv(session['csv_path'])
    df       = clean_df(df)
    numerics = df.select_dtypes(include=np.number).columns.tolist()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['SME MARKETING INTELLIGENCE — AI ANALYSIS REPORT'])
    writer.writerow(['Generated by: Scalable AI Framework | KIU University | ID: 15018'])
    writer.writerow(['File Analysed:', session.get('csv_name','')])
    writer.writerow(['Total Rows:', len(df)])
    writer.writerow(['Total Columns:', len(df.columns)])
    writer.writerow([])

    # Stats summary
    writer.writerow(['COLUMN STATISTICS'])
    writer.writerow(['Column','Mean','Max','Min','Std Dev'])
    for col in numerics[:8]:
        writer.writerow([
            col,
            round(float(df[col].mean()), 3),
            round(float(df[col].max()),  3),
            round(float(df[col].min()),  3),
            round(float(df[col].std()),  3),
        ])
    writer.writerow([])

    # Top performers
    if len(numerics) >= 1:
        target = numerics[0]
        writer.writerow([f'TOP 5 ROWS BY {target}'])
        top5 = df.nlargest(5, target)
        writer.writerow(top5.columns.tolist())
        for _, row in top5.iterrows():
            writer.writerow(row.tolist())
        writer.writerow([])

        writer.writerow([f'BOTTOM 5 ROWS BY {target}'])
        bot5 = df.nsmallest(5, target)
        writer.writerow(bot5.columns.tolist())
        for _, row in bot5.iterrows():
            writer.writerow(row.tolist())

    output.seek(0)
    filename = session.get('csv_name', 'results').replace('.csv','')

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename={filename}_Summary_Report.csv'}
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)