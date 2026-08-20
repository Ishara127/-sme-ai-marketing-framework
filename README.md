# 🎯 Scalable AI Framework for Digital Marketing Adoption in SMEs

**Final Year Research Project — COM4901**
**KIU University | Faculty of Computer Science and Engineering**

---

## 👩‍💻 Student Details
- **Name:** A.S.D. Ishara Lakmali Gunathilaka
- **Student ID:** 15018
- **Internal Supervisor:** Mr. Ravindra Withanage
- **External Supervisor:** Mr. Tharindu De Zoysa

---

## 📋 Project Overview
This research develops a **Scalable Artificial Intelligence Framework**
that enables Small and Medium Enterprises (SMEs) to adopt AI-driven
digital marketing practices in a modular, cost-effective, and
technically accessible manner.

### 🔍 Research Problem
Sri Lanka has 1.1 million SMEs contributing 52% of GDP, yet only
23% have implemented any AI solutions (ICTA, 2025). Enterprise AI
tools like Salesforce Einstein ($75+/month) are financially
inaccessible. This research addresses that gap.

---

## 🤖 AI Models Developed

| Model | Algorithm | Result |
|-------|-----------|--------|
| ROI Prediction | Linear Regression + Random Forest | R² = -0.0023 |
| Customer Segmentation | K-Means Clustering | 8 segments, Silhouette = 0.1255 |
| Campaign Classifier | Gradient Boosting (Best) | Accuracy = 42.25% |
| Algorithm Comparison | RF vs GB vs XGBoost | GB selected as best |
| Scalability Test | Sensitivity Analysis | Stable: 500–50,000 rows |

---

## ✅ Expert Validation Results (n=5)

| Dimension | Score |
|-----------|-------|
| Framework Evaluation | 4.84 / 5.00 (96.8%) |
| Web Application Usability | 5.00 / 5.00 (100%) |
| Research Quality | 4.33 / 5.00 (86.6%) |
| **Overall** | **4.72 / 5.00 (94.4%)** |

100% of respondents rated prototype as **"Excellent — production-ready"**

---

## 🌐 Web Application Features
- ✅ Home landing page
- ✅ CSV file upload (any marketing dataset)
- ✅ Automated data cleaning & preprocessing
- ✅ AI-powered customer segmentation
- ✅ Feature importance analysis
- ✅ Export results (CSV download)
- ✅ Demo dashboard (4 pages)
- ✅ Research mode + SME upload mode

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python Flask |
| AI/ML | Scikit-learn, XGBoost, Pandas, NumPy |
| Frontend | HTML5, CSS3, Chart.js |
| Dataset | Kaggle — Marketing Campaign Performance Dataset |
| Environment | Python venv |

---

## 📁 Project Structure


---

## 🚀 How to Run

```bash
# 1. Clone repository
git clone https://github.com/Ishara127/-sme-ai-marketing-framework

# 2. Navigate to folder
cd -sme-ai-marketing-framework

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn xgboost matplotlib seaborn

# 5. Run web application
cd webapp
python app.py

# 6. Open browser
# http://127.0.0.1:5000
```

---

## 📊 Key Research Findings

1. **ROI Predictability:** Structured campaign metrics alone cannot
   reliably predict SME marketing ROI — external factors dominate

2. **Customer Segments:** 8 distinct segments identified — Dormant
   segment shows highest ROI potential (6.60x avg)

3. **Best Classifier:** Gradient Boosting (42.25%) outperforms
   Random Forest (36.90%) and XGBoost (36.95%)

4. **Scalability Proven:** Consistent performance from 500 to
   50,000 rows — validates SME scalability claim

5. **Expert Validation:** Overall score 4.72/5.00 (94.4%) from
   5 domain experts — 100% rated as production-ready

---

## 📚 Dataset
- **Source:** Kaggle — Marketing Campaign Performance Dataset
- **Size:** 200,000 rows × 16 columns
- **Missing Values:** 0

---

## 📜 License
This project is developed for academic research purposes under
KIU University COM4901 module.

© 2026 A.S.D. Ishara Lakmali Gunathilaka — KIU University
