import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from src.preprocessing import load_data, engineer_features, transform_sample
from src.explainability import get_feature_importances, generate_reason_codes, compute_employee_shap_impact

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Palo Alto Networks | HR Attrition Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism & Cyber Dark Palette CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #E6E8EC;
    }
    
    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #1F2430 0%, #111319 100%);
        border: 1px solid rgba(255, 75, 75, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 30px;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #A0A5B5;
        font-size: 15px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(26, 31, 44, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    .metric-val {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 4px;
    }
    .metric-lbl {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Risk Category Badges */
    .badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-med {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    /* Reason Box */
    .reason-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #FF4B4B;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .reason-title {
        font-size: 16px;
        font-weight: 600;
        color: #F8FAFC;
    }
    .reason-desc {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
    }
    .reason-rec {
        font-size: 13px;
        color: #38BDF8;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Tabs Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 0px 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94A3B8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 75, 75, 0.05) 100%) !important;
        border-color: #FF4B4B !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    df = load_data('data/employee_attrition.csv')
    return df

@st.cache_resource
def load_model_artifacts():
    best_model = joblib.load('models/best_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    with open('models/evaluation_summary.json', 'r') as f:
        eval_summary = json.load(f)
        
    return best_model, preprocessor, feature_names, eval_summary

df_raw = load_dataset()
best_model, preprocessor, feature_names, eval_summary = load_model_artifacts()

# Compute employee-level probabilities across entire dataset for dashboard metrics
@st.cache_data
def compute_all_predictions(_model, _preprocessor, _df_raw, _feature_names):
    X_trans = transform_sample(_df_raw, _preprocessor, _feature_names)
    probs = _model.predict_proba(X_trans)[:, 1] if hasattr(_model, 'predict_proba') else _model.predict(X_trans)
    
    df_pred = _df_raw.copy()
    df_pred['Attrition_Probability'] = np.round(probs, 4)
    df_pred['Risk_Score_Pct'] = np.round(probs * 100, 1)
    
    conditions = [
        (df_pred['Attrition_Probability'] < 0.30),
        (df_pred['Attrition_Probability'] >= 0.30) & (df_pred['Attrition_Probability'] <= 0.60),
        (df_pred['Attrition_Probability'] > 0.60)
    ]
    categories = ['Low Risk', 'Medium Risk', 'High Risk']
    df_pred['Risk_Category'] = np.select(conditions, categories, default='Low Risk')
    df_pred['Employee_ID'] = [f"PAN-{1000 + i}" for i in range(len(df_pred))]
    return df_pred

df_scored = compute_all_predictions(best_model, preprocessor, df_raw, feature_names)

# -----------------------------------------------------------------------------
# 3. Sidebar Filtering & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.title("Palo Alto Networks")
    st.caption("HR Workforce Analytics & Attrition Intelligence")
    st.divider()

    st.subheader("📌 Global Filters")
    selected_dept = st.multiselect("Department", options=df_scored['Department'].unique(), default=df_scored['Department'].unique())
    selected_role = st.multiselect("Job Role", options=df_scored['JobRole'].unique(), default=df_scored['JobRole'].unique())
    selected_travel = st.multiselect("Business Travel", options=df_scored['BusinessTravel'].unique(), default=df_scored['BusinessTravel'].unique())
    
    st.divider()
    st.subheader("⚙️ Risk Threshold Sliders")
    low_thresh = st.slider("Low/Medium Risk Boundary (%)", 10, 40, 30)
    high_thresh = st.slider("Medium/High Risk Boundary (%)", 45, 80, 60)

# Filtered dataset view
filtered_df = df_scored[
    (df_scored['Department'].isin(selected_dept)) &
    (df_scored['JobRole'].isin(selected_role)) &
    (df_scored['BusinessTravel'].isin(selected_travel))
].copy()

# Recalculate dynamic categories based on user slider thresholds
filtered_df['Dynamic_Category'] = np.select(
    [
        (filtered_df['Risk_Score_Pct'] < low_thresh),
        (filtered_df['Risk_Score_Pct'] >= low_thresh) & (filtered_df['Risk_Score_Pct'] <= high_thresh),
        (filtered_df['Risk_Score_Pct'] > high_thresh)
    ],
    ['Low Risk', 'Medium Risk', 'High Risk'],
    default='Low Risk'
)

# -----------------------------------------------------------------------------
# 4. Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">Machine Learning–Based Employee Attrition & Risk Scoring System</div>
    <div class="header-subtitle">Predictive Decision Intelligence for Targeted HR Retention & Data-Driven Workforce Planning at Palo Alto Networks</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Attrition Risk Overview",
    "👤 Employee Risk Profile",
    "🏢 Department Intelligence",
    "🧠 Model Explainability",
    "🧪 What-If Simulator"
])

# -----------------------------------------------------------------------------
# TAB 1: Executive Attrition Risk Overview
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Executive Workforce Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Workforce</div>
            <div class="metric-val">{len(filtered_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        attrition_rate = (filtered_df['Attrition'] == 1).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Historical Attrition Rate</div>
            <div class="metric-val" style="color: #FF4B4B;">{attrition_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        high_risk_count = (filtered_df['Dynamic_Category'] == 'High Risk').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">High Risk Employees</div>
            <div class="metric-val" style="color: #EF4444;">{high_risk_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_risk = filtered_df['Risk_Score_Pct'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Avg Risk Probability</div>
            <div class="metric-val" style="color: #F59E0B;">{avg_risk:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        overtime_pct = (filtered_df['OverTime'] == 'Yes').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">OverTime Workforce</div>
            <div class="metric-val" style="color: #38BDF8;">{overtime_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 🎯 Attrition Risk Category Distribution")
        risk_counts = filtered_df['Dynamic_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Category', 'Count']
        
        fig_donut = px.pie(
            risk_counts,
            names='Risk_Category',
            values='Count',
            hole=0.55,
            color='Risk_Category',
            color_discrete_map={'Low Risk': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E8EC',
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("#### 🏢 Department-Level Risk Breakdown")
        dept_risk = filtered_df.groupby('Department')['Risk_Score_Pct'].mean().reset_index().sort_values('Risk_Score_Pct', ascending=True)
        
        fig_dept = px.bar(
            dept_risk,
            x='Risk_Score_Pct',
            y='Department',
            orientation='h',
            text='Risk_Score_Pct',
            color='Risk_Score_Pct',
            color_continuous_scale='Reds'
        )
        fig_dept.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dept.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E8EC',
            xaxis_title="Average Attrition Risk Score (%)",
            yaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    st.divider()
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### ⏳ OverTime Impact on Attrition Risk")
        ot_df = filtered_df.groupby(['OverTime', 'Dynamic_Category']).size().reset_index(name='Count')
        fig_ot = px.bar(
            ot_df,
            x='OverTime',
            y='Count',
            color='Dynamic_Category',
            barmode='group',
            color_discrete_map={'Low Risk': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
        )
        fig_ot.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E8EC',
            xaxis_title="Works OverTime",
            yaxis_title="Employee Count"
        )
        st.plotly_chart(fig_ot, use_container_width=True)
        
    with col_b:
        st.markdown("#### 💰 Salary & Experience vs Attrition Risk")
        fig_scatter = px.scatter(
            filtered_df,
            x='TotalWorkingYears',
            y='MonthlyIncome',
            color='Dynamic_Category',
            size='Risk_Score_Pct',
            hover_data=['Employee_ID', 'JobRole', 'YearsAtCompany'],
            color_discrete_map={'Low Risk': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E8EC',
            xaxis_title="Total Working Years",
            yaxis_title="Monthly Income ($)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Employee Risk Profile & Reason Codes
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 👤 Individual Employee Risk Assessment & Reason Codes")
    
    col_select, col_info = st.columns([1, 2])
    
    with col_select:
        emp_list = filtered_df['Employee_ID'].tolist()
        selected_emp_id = st.selectbox("🔍 Select Employee ID", options=emp_list)
        
        emp_row = filtered_df[filtered_df['Employee_ID'] == selected_emp_id].iloc[0]
        risk_pct = emp_row['Risk_Score_Pct']
        cat = emp_row['Dynamic_Category']
        
        # Risk Badge display
        if cat == 'High Risk':
            badge_html = f'<div class="badge-high">🚨 HIGH ATTRITION RISK ({risk_pct:.1f}%)</div>'
        elif cat == 'Medium Risk':
            badge_html = f'<div class="badge-med">⚠️ MEDIUM ATTRITION RISK ({risk_pct:.1f}%)</div>'
        else:
            badge_html = f'<div class="badge-low">✅ LOW ATTRITION RISK ({risk_pct:.1f}%)</div>'
            
        st.markdown(f"<br>{badge_html}<br>", unsafe_allow_html=True)
        
        # Attrition Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_pct,
            number = {'suffix': "%", 'font': {'size': 28, 'color': "#FFFFFF"}},
            title = {'text': "Attrition Probability", 'font': {'size': 14, 'color': "#94A3B8"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': "#FF4B4B" if risk_pct > 60 else "#F59E0B" if risk_pct >= 30 else "#10B981"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                    {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E8EC',
            height=200,
            margin=dict(t=30, b=10, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.markdown(f"#### Employee Details: **{selected_emp_id}**")
        
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Department:** {emp_row['Department']}")
        c1.write(f"**Job Role:** {emp_row['JobRole']}")
        c1.write(f"**Age:** {emp_row['Age']} yrs")
        
        c2.write(f"**Monthly Income:** ${emp_row['MonthlyIncome']:,}")
        c2.write(f"**OverTime:** {emp_row['OverTime']}")
        c2.write(f"**Years at Company:** {emp_row['YearsAtCompany']} yrs")

        c3.write(f"**Job Level:** {emp_row['JobLevel']}")
        c3.write(f"**Marital Status:** {emp_row['MaritalStatus']}")
        c3.write(f"**Last Promotion:** {emp_row['YearsSinceLastPromotion']} yrs ago")
        
        st.divider()
        st.markdown("#### 💡 Top Contributing Risk Factors & Reason Codes")
        reasons = generate_reason_codes(emp_row)
        
        for r in reasons:
            severity_color = "#EF4444" if r['Severity'] == 'High' else "#F59E0B" if r['Severity'] == 'Medium' else "#10B981"
            st.markdown(f"""
            <div class="reason-box" style="border-left-color: {severity_color};">
                <div class="reason-title">{r['Factor']} <span style="font-size: 12px; color: {severity_color};">[{r['Severity']} Severity]</span></div>
                <div class="reason-desc">{r['Detail']}</div>
                <div class="reason-rec">👉 <strong>HR Action Recommendation:</strong> {r['Recommendation']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 🔬 Individual Feature Impact Breakdown")
    sample_df_single = pd.DataFrame([emp_row.to_dict()])
    impact_df = compute_employee_shap_impact(best_model, preprocessor, sample_df_single, feature_names)
    
    fig_impact = px.bar(
        impact_df,
        x='Impact',
        y='Feature',
        orientation='h',
        color='Impact',
        color_continuous_scale='RdYlGn_r'
    )
    fig_impact.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E6E8EC',
        xaxis_title="Contribution to Attrition Risk Probability",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_impact, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Department & Role Intelligence Matrix
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🏢 Cross-Department & Job Role Risk Matrix")
    
    heatmap_df = filtered_df.groupby(['Department', 'JobRole'])['Risk_Score_Pct'].mean().unstack().fillna(0)
    
    fig_heat = px.imshow(
        heatmap_df,
        labels=dict(x="Job Role", y="Department", color="Avg Risk (%)"),
        x=heatmap_df.columns,
        y=heatmap_df.index,
        color_continuous_scale='Reds',
        text_auto='.1f'
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E6E8EC',
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.divider()
    st.markdown("### 📋 Actionable High-Risk Employee Roster")
    
    high_risk_roster = filtered_df[filtered_df['Dynamic_Category'].isin(['High Risk', 'Medium Risk'])].sort_values('Risk_Score_Pct', ascending=False)
    
    st.dataframe(
        high_risk_roster[['Employee_ID', 'Department', 'JobRole', 'Age', 'MonthlyIncome', 'OverTime', 'YearsSinceLastPromotion', 'Risk_Score_Pct', 'Dynamic_Category']],
        use_container_width=True,
        hide_index=True
    )
    
    csv_data = high_risk_roster.to_csv(index=False)
    st.download_button(
        label="📥 Download High-Risk Employee Roster (CSV)",
        data=csv_data,
        file_name="Palo_Alto_Networks_High_Risk_Employees.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# TAB 4: Model Explainability & Analytics
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🧠 Machine Learning Model Explainability & Performance Benchmarks")
    
    col_bench, col_imp = st.columns([1, 1])
    
    with col_bench:
        st.markdown("#### 📊 Model Evaluation Benchmark Table")
        metrics_df = pd.DataFrame(eval_summary['metrics']).T
        st.dataframe(metrics_df, use_container_width=True)
        st.success(f"⭐ **Current Production Model**: {eval_summary['best_model_name']}")

    with col_imp:
        st.markdown("#### 🌟 Global Top Feature Importances")
        df_imp = get_feature_importances(best_model, feature_names).head(12)
        fig_imp = px.bar(
            df_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Reds'
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E8EC',
            xaxis_title="Global Model Weight / Importance",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: What-If Scenario Simulator
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 🧪 What-If HR Retention Scenario Simulator")
    st.caption("Simulate targeted retention interventions (e.g. salary hikes, overtime elimination, work-life balance adjustments) and calculate real-time attrition risk reduction.")
    
    col_sim_select, col_sim_controls = st.columns([1, 2])
    
    with col_sim_select:
        sim_emp_id = st.selectbox("Select Employee to Simulate", options=filtered_df['Employee_ID'].tolist(), key='sim_emp_select')
        sim_emp_row = filtered_df[filtered_df['Employee_ID'] == sim_emp_id].iloc[0].to_dict()
        
        orig_prob = sim_emp_row['Attrition_Probability']
        orig_pct = sim_emp_row['Risk_Score_Pct']
        orig_cat = sim_emp_row['Dynamic_Category']
        
        st.markdown("#### Baseline Profile")
        st.write(f"**Baseline Risk:** {orig_pct:.1f}% ({orig_cat})")
        st.write(f"**Monthly Income:** ${sim_emp_row['MonthlyIncome']:,}")
        st.write(f"**OverTime:** {sim_emp_row['OverTime']}")
        st.write(f"**Work-Life Balance:** {sim_emp_row['WorkLifeBalance']}/4")

    with col_sim_controls:
        st.markdown("#### 🎛️ Intervention Parameters")
        
        sim_salary_hike = st.slider("Salary Increase (%)", 0, 50, 15)
        sim_overtime = st.radio("OverTime Status", options=['No', 'Yes'], index=0 if sim_emp_row['OverTime']=='Yes' else 1)
        sim_wlb = st.slider("Target Work-Life Balance (1-4)", 1, 4, max(3, int(sim_emp_row['WorkLifeBalance'])))
        sim_job_sat = st.slider("Target Job Satisfaction (1-4)", 1, 4, max(3, int(sim_emp_row['JobSatisfaction'])))

        # Create modified profile
        mod_profile = sim_emp_row.copy()
        mod_profile['MonthlyIncome'] = int(sim_emp_row['MonthlyIncome'] * (1 + sim_salary_hike / 100))
        mod_profile['OverTime'] = sim_overtime
        mod_profile['WorkLifeBalance'] = sim_wlb
        mod_profile['JobSatisfaction'] = sim_job_sat
        
        # Predict modified sample risk
        mod_df = pd.DataFrame([mod_profile])
        X_mod = transform_sample(mod_df, preprocessor, feature_names)
        mod_prob = best_model.predict_proba(X_mod)[0, 1] if hasattr(best_model, 'predict_proba') else best_model.predict(X_mod)[0]
        mod_pct = round(mod_prob * 100, 1)
        
        mod_cat = 'Low Risk' if mod_pct < low_thresh else 'Medium Risk' if mod_pct <= high_thresh else 'High Risk'
        
        risk_diff = orig_pct - mod_pct
        
        st.divider()
        st.markdown("#### 🏁 Post-Intervention Simulated Results")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Risk Score", f"{mod_pct:.1f}%", delta=f"-{risk_diff:.1f}%", delta_color="normal")
        m2.metric("Simulated Risk Category", mod_cat)
        m3.metric("Simulated Monthly Income", f"${mod_profile['MonthlyIncome']:,}", delta=f"+{sim_salary_hike}%")
        
        if risk_diff > 0:
            st.success(f"✨ Intervention successful! Attrition risk reduced by **{risk_diff:.1f}%**.")
        else:
            st.info("Adjust intervention parameters to see potential risk reduction.")
