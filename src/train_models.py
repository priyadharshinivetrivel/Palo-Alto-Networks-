import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from imblearn.over_sampling import SMOTE

from src.preprocessing import load_data, fit_transform_pipeline

def train_and_evaluate_models():
    """
    Executes model training pipeline:
    - Preprocessing & Feature Engineering
    - Train-Test Split (Stratified)
    - SMOTE Imbalance Handling
    - Multi-Model Benchmarking (Logistic Regression, Random Forest, XGBoost, LightGBM)
    - Saves trained artifacts to models/ directory
    """
    os.makedirs('models', exist_ok=True)
    
    print("Loading data...")
    df_raw = load_data('data/employee_attrition.csv')
    
    print("Applying preprocessing & feature engineering...")
    X, y, preprocessor, feature_names = fit_transform_pipeline(df_raw)
    
    # Save preprocessor and feature names
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Train-test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Handle Class Imbalance using SMOTE
    print("Applying SMOTE resampling on training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Define models dictionary
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, verbose=-1)
    }
    
    results = {}
    best_roc_auc = -1.0
    best_model_name = None
    best_model_obj = None
    
    print("\n--- Model Training & Benchmarking ---")
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': float(round(acc, 4)),
            'Precision': float(round(prec, 4)),
            'Recall': float(round(rec, 4)),
            'F1-Score': float(round(f1, 4)),
            'ROC-AUC': float(round(roc_auc, 4)),
            'ConfusionMatrix': cm
        }
        
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        # Save individual model
        filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model_obj = model

    print(f"\nBest Model by ROC-AUC: {best_model_name} ({best_roc_auc:.4f})")
    joblib.dump(best_model_obj, 'models/best_model.pkl')
    
    summary_data = {
        'best_model_name': best_model_name,
        'metrics': results
    }
    
    with open('models/evaluation_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print("Model training complete. Artifacts saved to models/")

if __name__ == '__main__':
    train_and_evaluate_models()
