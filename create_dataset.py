import os
import pandas as pd
import numpy as np

os.makedirs('data', exist_ok=True)
csv_path = 'data/employee_attrition.csv'

np.random.seed(42)
n_samples = 1470

departments = ['Research & Development', 'Sales', 'Human Resources']
dept_prob = [0.65, 0.30, 0.05]

job_roles = {
    'Research & Development': ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Research Director', 'Manager'],
    'Sales': ['Sales Executive', 'Sales Representative', 'Manager'],
    'Human Resources': ['Human Resources', 'Manager']
}

education_fields = {
    'Research & Development': ['Life Sciences', 'Medical', 'Technical Degree', 'Other'],
    'Sales': ['Marketing', 'Life Sciences', 'Medical', 'Other'],
    'Human Resources': ['Human Resources', 'Life Sciences', 'Medical', 'Other']
}

data = []
for i in range(n_samples):
    age = int(np.random.randint(20, 60))
    dept = np.random.choice(departments, p=dept_prob)
    role = np.random.choice(job_roles[dept])
    edu_field = np.random.choice(education_fields[dept])
    
    gender = np.random.choice(['Male', 'Female'], p=[0.6, 0.4])
    marital = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.32, 0.46, 0.22])
    travel = np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], p=[0.10, 0.70, 0.20])
    overtime = np.random.choice(['Yes', 'No'], p=[0.30, 0.70])
    
    distance = int(np.random.randint(1, 30))
    education = int(np.random.choice([1, 2, 3, 4, 5], p=[0.11, 0.19, 0.39, 0.27, 0.04]))
    
    env_sat = int(np.random.choice([1, 2, 3, 4], p=[0.20, 0.20, 0.30, 0.30]))
    job_inv = int(np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.50, 0.15]))
    job_sat = int(np.random.choice([1, 2, 3, 4], p=[0.20, 0.20, 0.30, 0.30]))
    rel_sat = int(np.random.choice([1, 2, 3, 4], p=[0.20, 0.20, 0.30, 0.30]))
    wlb = int(np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.55, 0.10]))
    
    if role in ['Manager', 'Research Director']:
        job_level = int(np.random.choice([4, 5], p=[0.6, 0.4]))
    elif role in ['Healthcare Representative', 'Manufacturing Director', 'Sales Executive']:
        job_level = int(np.random.choice([2, 3], p=[0.7, 0.3]))
    elif role in ['Sales Representative']:
        job_level = 1
    else:
        job_level = int(np.random.choice([1, 2], p=[0.6, 0.4]))
        
    tot_work_years = max(job_level * 2, int(min(age - 18, np.random.randint(job_level * 2, job_level * 5 + 1))))
    years_at_co = min(tot_work_years, int(np.random.randint(1, max(2, tot_work_years + 1))))
    years_curr_role = min(years_at_co, int(np.random.randint(0, years_at_co + 1)))
    years_promo = min(years_at_co, int(np.random.randint(0, years_at_co + 1)))
    years_mgr = min(years_at_co, int(np.random.randint(0, years_at_co + 1)))
    
    num_comp = int(np.random.randint(0, 9))
    stock_level = int(np.random.choice([0, 1, 2, 3], p=[0.44, 0.40, 0.10, 0.06]))
    
    base_income = 2000 + (job_level * 3000) + (tot_work_years * 100)
    monthly_income = int(max(1200, min(20000, base_income + np.random.randint(-500, 500))))
    daily_rate = int(np.random.randint(100, 1500))
    hourly_rate = int(np.random.randint(30, 100))
    monthly_rate = int(np.random.randint(2000, 27000))
    
    percent_hike = int(np.random.randint(11, 26))
    perf_rating = 4 if percent_hike >= 20 else 3
    training_last_year = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.05, 0.05, 0.35, 0.35, 0.08, 0.07, 0.05]))
    
    # Deterministic risk logit formulation with low noise
    logit = (
        + 1.8 * (overtime == 'Yes')
        + 1.2 * (job_sat <= 2)
        + 1.1 * (env_sat <= 2)
        + 1.3 * (wlb <= 2)
        + 0.9 * (marital == 'Single')
        + 0.8 * (distance > 15)
        + 1.0 * (stock_level == 0)
        + 0.9 * (years_promo >= 4)
        + 0.7 * (travel == 'Travel_Frequently')
        - 1.2 * (job_level >= 3)
        - 1.0 * (monthly_income > 8000)
        - 0.8 * (years_with_mgr := years_mgr > 4)
        - 3.2
    )
    
    prob = 1 / (1 + np.exp(-logit))
    attrition = 1 if np.random.rand() < prob else 0

    data.append({
        'Age': age,
        'Attrition': attrition,
        'BusinessTravel': travel,
        'DailyRate': daily_rate,
        'Department': dept,
        'DistanceFromHome': distance,
        'Education': education,
        'EducationField': edu_field,
        'EnvironmentSatisfaction': env_sat,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_inv,
        'JobLevel': job_level,
        'JobRole': role,
        'JobSatisfaction': job_sat,
        'MaritalStatus': marital,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_comp,
        'OverTime': overtime,
        'PercentSalaryHike': percent_hike,
        'PerformanceRating': perf_rating,
        'RelationshipSatisfaction': rel_sat,
        'StockOptionLevel': stock_level,
        'TotalWorkingYears': tot_work_years,
        'TrainingTimesLastYear': training_last_year,
        'WorkLifeBalance': wlb,
        'YearsAtCompany': years_at_co,
        'YearsInCurrentRole': years_curr_role,
        'YearsSinceLastPromotion': years_promo,
        'YearsWithCurrManager': years_mgr
    })

df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)
print(f"Generated Palo Alto Networks HR Attrition Dataset at {csv_path} with {len(df)} records.")
print(f"Attrition count:\n{df['Attrition'].value_counts(normalize=True)}")
