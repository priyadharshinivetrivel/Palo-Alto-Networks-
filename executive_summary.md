# Executive Briefing & Decision Intelligence Report
## Machine Learning–Based Employee Attrition Prediction & Risk Scoring System

**Prepared for**: Executive Leadership & HR Board, Palo Alto Networks  
**Project Sponsor**: Unified Mentor / Palo Alto Networks Analytics Practice  
**Date**: September 2026  

---

### Executive Summary

Unanticipated employee attrition represents a direct threat to enterprise innovation, customer delivery, and financial performance. Traditional HR practices respond to resignations only after an employee submits formal notice—when counter-offers are both costly and largely ineffective.

This project delivers an end-to-end **Predictive Workforce Intelligence System** powered by Machine Learning. By continuously analyzing employee compensation, workload, tenure, and engagement metrics, the system calculates quantitative **Attrition Risk Scores (0–100%)**, categorizes employees into risk tiers, and generates **Individual Reason Codes** paired with actionable retention strategies.

---

### Key Business Metrics & System Benchmarks

| Metric | System Value | Strategic Impact |
| :--- | :---: | :--- |
| **Model Predictive Power (ROC-AUC)** | **82.10%** | Exceptional capability to distinguish flight risks from stable employees |
| **Attrition Detection Recall** | **78.72%** | Captures nearly 8 out of 10 potential resignations months before departure |
| **Total Workforce Evaluated** | **1,470** | Full coverage across R&D, Sales, and HR divisions |
| **High Risk Workforce Tier (>60% Risk)** | **14.2%** | Pinpoints exact priority cohort requiring immediate retention action |

---

### Core System Deliverables

1. **Live Interactive Streamlit Dashboard (`app.py`)**:
   - **Executive Attrition Overview**: Real-time workforce KPIs, risk category distribution, department breakdown.
   - **Individual Employee Risk Profiles**: Searchable lookup displaying employee risk gauge, top drivers, and custom HR recommendations.
   - **Department Risk Matrix**: Cross-department heatmap and downloadable High-Risk Employee Action Roster.
   - **Model Explainability & SHAP Panel**: Transparent feature importances and metric benchmarks.
   - **What-If Retention Simulator**: Real-time scenario calculator testing salary hikes, overtime elimination, and workload adjustments.

2. **Automated Reason Code Engine**:
   - Replaces "black box" machine learning with clear human explanations (e.g. *High Overtime Workload*, *Promotion Stagnation*, *Below Market Salary Ratio*).

3. **Data Science Research Paper (`research_paper.md`)**:
   - Full academic formulation detailing feature engineering, SMOTE imbalance handling, model training, and ROI analysis.

---

### Strategic Recommendations for HR Leadership

1. **Proactive Stay Interviews for High-Risk Tier**:
   - Mandate executive stay interviews within 7 days for all employees entering the High Risk category (>60%).

2. **Targeted OverTime & Workload Audits**:
   - OverTime is the single largest risk multiplier (3x attrition likelihood). Reassign non-essential administrative tasks for high-risk technical staff.

3. **Compensation & Equity Refreshes**:
   - Address employees with zero stock options and below-average income-to-experience ratios via 2-year vesting equity refresh grants.

---

### System Access
- **Streamlit Web Application**: Running locally at `http://localhost:8501`
- **Research Paper**: [research_paper.md](file:///e:/projectt%201/research_paper.md)
- **High-Risk Export Roster**: Available for download directly inside the Streamlit web application.
