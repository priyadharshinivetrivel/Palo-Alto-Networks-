import unittest
import os
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import load_data, engineer_features, fit_transform_pipeline, transform_sample
from src.explainability import get_feature_importances, generate_reason_codes, compute_employee_shap_impact

class TestAttritionPipeline(unittest.TestCase):

    def setUp(self):
        self.csv_path = 'data/employee_attrition.csv'
        self.assertTrue(os.path.exists(self.csv_path), "Dataset file should exist")
        self.df = load_data(self.csv_path)

    def test_feature_engineering(self):
        df_feat = engineer_features(self.df)
        self.assertIn('Income_to_Experience_Ratio', df_feat.columns)
        self.assertIn('Promotion_Delay_Ratio', df_feat.columns)
        self.assertIn('Role_Tenure_Ratio', df_feat.columns)
        self.assertIn('Engagement_Score', df_feat.columns)
        self.assertIn('Workload_Stress_Flag', df_feat.columns)
        self.assertEqual(len(df_feat), len(self.df))

    def test_pipeline_transform(self):
        X_trans, y, preprocessor, feature_names = fit_transform_pipeline(self.df)
        self.assertIsNotNone(X_trans)
        self.assertIsNotNone(y)
        self.assertGreater(len(feature_names), 20)
        self.assertEqual(len(X_trans), len(self.df))

    def test_model_artifacts_exist(self):
        self.assertTrue(os.path.exists('models/best_model.pkl'))
        self.assertTrue(os.path.exists('models/preprocessor.pkl'))
        self.assertTrue(os.path.exists('models/feature_names.pkl'))
        self.assertTrue(os.path.exists('models/evaluation_summary.json'))

    def test_reason_codes_generation(self):
        sample_high_risk = {
            'OverTime': 'Yes',
            'JobSatisfaction': 1,
            'WorkLifeBalance': 1,
            'YearsSinceLastPromotion': 5,
            'YearsAtCompany': 6,
            'StockOptionLevel': 0,
            'MonthlyIncome': 2500,
            'TotalWorkingYears': 10,
            'DistanceFromHome': 25
        }
        reasons = generate_reason_codes(sample_high_risk)
        self.assertGreater(len(reasons), 2)
        factors = [r['Factor'] for r in reasons]
        self.assertIn('High OverTime Workload', factors)
        self.assertIn('Low Job Satisfaction', factors)

if __name__ == '__main__':
    unittest.main()
