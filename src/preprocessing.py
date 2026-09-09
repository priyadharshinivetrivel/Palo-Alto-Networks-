import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def load_data(filepath='data/employee_attrition.csv'):
    """Loads dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    return pd.read_csv(filepath)

def engineer_features(df):
    """
    Applies custom feature engineering calculations specified in data science methodology:
    1. Income-to-experience ratio
    2. Promotion delay ratio
    3. Role tenure ratio
    4. Engagement composite score
    5. Workload stress flag
    """
    df_feat = df.copy()
    
    # 1. Income-to-experience ratio
    df_feat['Income_to_Experience_Ratio'] = df_feat['MonthlyIncome'] / (df_feat['TotalWorkingYears'] + 1)
    
    # 2. Promotion delay ratio
    df_feat['Promotion_Delay_Ratio'] = df_feat['YearsSinceLastPromotion'] / (df_feat['YearsAtCompany'] + 1)
    
    # 3. Role tenure ratio
    df_feat['Role_Tenure_Ratio'] = df_feat['YearsInCurrentRole'] / (df_feat['YearsAtCompany'] + 1)
    
    # 4. Engagement composite score (Mean of satisfaction metrics 1-4 scale)
    satisfaction_cols = ['JobInvolvement', 'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    df_feat['Engagement_Score'] = df_feat[satisfaction_cols].mean(axis=1)
    
    # 5. Workload stress flag (Binary indicator for high workload / low balance / commute distance)
    is_overtime = (df_feat['OverTime'] == 'Yes') | (df_feat['OverTime'] == 1)
    low_wlb = df_feat['WorkLifeBalance'] <= 2
    far_commute = df_feat['DistanceFromHome'] > 15
    low_env_sat = df_feat['EnvironmentSatisfaction'] <= 2
    
    df_feat['Workload_Stress_Flag'] = (is_overtime & (low_wlb | far_commute | low_env_sat)).astype(int)
    
    return df_feat

def get_feature_lists(df):
    """Returns numerical and categorical feature column names."""
    drop_cols = ['Attrition']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    cat_cols = [c for c in cat_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    return feature_cols, num_cols, cat_cols

def fit_transform_pipeline(df):
    """
    Fits and transforms raw dataset dataframe into preprocessed feature matrix X and target y.
    Returns X_transformed, y, preprocessor, feature_names
    """
    df_engineered = engineer_features(df)
    
    if 'Attrition' in df_engineered.columns:
        if df_engineered['Attrition'].dtype == object:
            y = (df_engineered['Attrition'] == 'Yes').astype(int)
        else:
            y = df_engineered['Attrition'].astype(int)
    else:
        y = None
        
    feature_cols, num_cols, cat_cols = get_feature_lists(df_engineered)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_raw = df_engineered[feature_cols]
    X_transformed = preprocessor.fit_transform(X_raw)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + encoded_cat_cols
    
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    return X_transformed_df, y, preprocessor, feature_names

def transform_sample(sample_df, preprocessor, feature_names):
    """Transforms a single employee sample or new dataframe using pre-fitted preprocessor."""
    sample_engineered = engineer_features(sample_df)
    feature_cols = preprocessor.feature_names_in_ if hasattr(preprocessor, 'feature_names_in_') else None
    
    X_trans = preprocessor.transform(sample_engineered)
    return pd.DataFrame(X_trans, columns=feature_names)
