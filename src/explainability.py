import pandas as pd
import numpy as np
import shap
import joblib
from src.preprocessing import engineer_features

def get_feature_importances(model, feature_names):
    """
    Extracts global feature importances or linear coefficients from a fitted model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return df_imp

def generate_reason_codes(raw_row):
    """
    Generates human-readable individual-level risk reason codes for HR leaders.
    Evaluates key employee risk flags and returns top contributing factors and recommended retention interventions.
    """
    reasons = []
    
    # 1. Overtime Workload Stress
    overtime_val = str(raw_row.get('OverTime', '')).strip()
    if overtime_val.lower() in ['yes', '1', 'true']:
        reasons.append({
            'Factor': 'High OverTime Workload',
            'Severity': 'High',
            'Detail': 'Employee regularly works overtime, causing workload stress.',
            'Recommendation': 'Conduct workload audit & transition non-core tasks to mitigate burnout.'
        })
        
    # 2. Satisfaction Metrics
    job_sat = int(raw_row.get('JobSatisfaction', 4))
    if job_sat <= 2:
        reasons.append({
            'Factor': 'Low Job Satisfaction',
            'Severity': 'High' if job_sat == 1 else 'Medium',
            'Detail': f'Job satisfaction rating is low ({job_sat}/4).',
            'Recommendation': 'Schedule 1-on-1 stay interview to discuss role alignment and growth.'
        })
        
    env_sat = int(raw_row.get('EnvironmentSatisfaction', 4))
    if env_sat <= 2:
        reasons.append({
            'Factor': 'Workplace Environment Friction',
            'Severity': 'Medium',
            'Detail': f'Environment satisfaction rating is low ({env_sat}/4).',
            'Recommendation': 'Assess team dynamics, hybrid flexibility, and office working conditions.'
        })

    wlb = int(raw_row.get('WorkLifeBalance', 4))
    if wlb <= 2:
        reasons.append({
            'Factor': 'Poor Work-Life Balance',
            'Severity': 'High' if wlb == 1 else 'Medium',
            'Detail': f'Work-life balance rating is poor ({wlb}/4).',
            'Recommendation': 'Offer flexible working hours, remote options, or mandatory downtime.'
        })
        
    # 3. Career Stagnation & Promotion Delay
    years_promo = int(raw_row.get('YearsSinceLastPromotion', 0))
    years_at_co = int(raw_row.get('YearsAtCompany', 1))
    if years_promo >= 4 and years_at_co >= 3:
        reasons.append({
            'Factor': 'Promotion Stagnation Delay',
            'Severity': 'High',
            'Detail': f'No promotion in the last {years_promo} years at Palo Alto Networks.',
            'Recommendation': 'Review career progression framework and evaluate promotion readiness.'
        })
        
    # 4. Financial & Stock Incentive Risk
    stock = int(raw_row.get('StockOptionLevel', 0))
    if stock == 0:
        reasons.append({
            'Factor': 'Zero Stock Option Equity',
            'Severity': 'Medium',
            'Detail': 'Employee holds Level 0 stock options (no long-term equity retention hook).',
            'Recommendation': 'Consider equity grant refresh or retention bonus tied to 2-year vesting.'
        })
        
    monthly_inc = float(raw_row.get('MonthlyIncome', 5000))
    tot_work_years = float(raw_row.get('TotalWorkingYears', 1))
    inc_exp_ratio = monthly_inc / (tot_work_years + 1)
    if inc_exp_ratio < 350 and monthly_inc < 4500:
        reasons.append({
            'Factor': 'Below Market Salary Ratio',
            'Severity': 'High',
            'Detail': f'Monthly income (${monthly_inc:,.0f}) is below benchmark for {tot_work_years:.0f} yrs experience.',
            'Recommendation': 'Perform out-of-cycle compensation market benchmark adjustment.'
        })

    # 5. Commute Distance
    distance = int(raw_row.get('DistanceFromHome', 0))
    if distance > 15:
        reasons.append({
            'Factor': 'Long Commute Distance',
            'Severity': 'Low',
            'Detail': f'Employee lives {distance} km from workplace.',
            'Recommendation': 'Provide hybrid / work-from-home stipend or commuting allowance.'
        })

    # Default fallback if no specific flags triggered
    if not reasons:
        reasons.append({
            'Factor': 'Standard Career Transition Check',
            'Severity': 'Low',
            'Detail': 'Employee exhibits balanced metrics across key risk drivers.',
            'Recommendation': 'Maintain quarterly check-ins and standard professional development.'
        })
        
    return reasons

def compute_employee_shap_impact(model, preprocessor, sample_df, feature_names):
    """
    Computes local feature contribution values for a given employee sample.
    """
    from src.preprocessing import transform_sample
    X_sample = transform_sample(sample_df, preprocessor, feature_names)
    
    if hasattr(model, 'coef_'):
        # Linear model explanation
        coefs = model.coef_[0]
        sample_vals = X_sample.values[0]
        impacts = coefs * sample_vals
    elif hasattr(model, 'feature_importances_'):
        # Tree model importance weighted sample
        importances = model.feature_importances_
        sample_vals = X_sample.values[0]
        impacts = importances * sample_vals
    else:
        impacts = np.zeros(len(feature_names))
        
    df_impact = pd.DataFrame({
        'Feature': feature_names,
        'Impact': impacts,
        'AbsImpact': np.abs(impacts)
    }).sort_values(by='AbsImpact', ascending=False).head(10).reset_index(drop=True)
    
    return df_impact
