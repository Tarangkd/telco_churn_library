# test_feature_engineering.py

import unittest
import pandas as pd
import numpy as np
from telco_churn.feature_engineering.feature_engineering import StatisticalFeatures, CategoricalEncoding, InteractionFeatures, TemporalFeatures, DerivedFeatures, FeaturePipeline

class TestFeatureEngineering(unittest.TestCase):

    def test_statistical_features(self):
        df = pd.DataFrame({'group': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40]})
        transformer = StatisticalFeatures(group_by_column='group')
        df_transformed = transformer.transform(df)
        self.assertIn('value_mean', df_transformed.columns, "StatisticalFeatures failed!")
        self.assertIn('value_std', df_transformed.columns, "StatisticalFeatures failed!")

    def test_categorical_encoding(self):
        df = pd.DataFrame({'category': ['a', 'b', 'a']})
        transformer = CategoricalEncoding()
        df_transformed = transformer.transform(df)
        self.assertIn('category', df_transformed.columns, "CategoricalEncoding failed!")
        self.assertTrue(df['category'].dtype == 'int64', "CategoricalEncoding failed to encode properly!")

    def test_interaction_features(self):
        df = pd.DataFrame({'value1': [1, 2, 3], 'value2': [4, 5, 6]})
        transformer = InteractionFeatures()
        df_transformed = transformer.transform(df)
        self.assertIn('value1_x_value2', df_transformed.columns, "InteractionFeatures failed!")

    def test_derived_features(self):
        df = pd.DataFrame({'tenure': [1, 2, 3], 'MonthlyCharges': [50, 60, 70], 'SeniorCitizen': [1, 0, 1]})
        transformer = DerivedFeatures()
        df_transformed = transformer.transform(df)
        self.assertIn('tenure_monthly_ratio', df_transformed.columns, "DerivedFeatures failed!")
        self.assertIn('is_senior', df_transformed.columns, "DerivedFeatures failed!")

    def test_temporal_features(self):
        df = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2022-02-01', '2023-03-01'])})
        transformer = TemporalFeatures()
        df_transformed = transformer.transform(df)
        self.assertIn('date_year', df_transformed.columns, "TemporalFeatures failed!")
        self.assertIn('date_month', df_transformed.columns, "TemporalFeatures failed!")
        self.assertIn('date_day', df_transformed.columns, "TemporalFeatures failed!")

    def test_feature_pipeline(self):
        df = pd.DataFrame({'group': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40], 'category': ['a', 'b', 'a', 'b']})
        pipeline = FeaturePipeline(transformers=[
            StatisticalFeatures(group_by_column='group'),
            CategoricalEncoding()
        ])
        df_transformed = pipeline.apply(df)
        self.assertIn('value_mean', df_transformed.columns, "Pipeline StatisticalFeatures failed!")
        self.assertIn('category', df_transformed.columns, "Pipeline CategoricalEncoding failed!")
        self.assertTrue(df_transformed['category'].dtype == 'int64', "Pipeline CategoricalEncoding failed to encode properly!")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
