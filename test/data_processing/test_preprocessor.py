import unittest
import pandas as pd
import numpy as np
import sys
import os
from telco_churn.data_preprocessing.preprocessor import HandleMissingValues,NormalizeData, EncodeCategoricalData, HandleOutliers,PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):

    def test_handle_missing_values(self):
        df = pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, 6]})
        preprocessor = HandleMissingValues(strategy='mean')
        df_transformed = preprocessor.process(df)
        self.assertFalse(df_transformed.isnull().any().any(), "HandleMissingValues failed!")

    def test_normalize_data(self):
        df = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessor = NormalizeData(method='zscore')
        df_transformed = preprocessor.process(df)
        self.assertTrue(np.isclose(df_transformed.mean().values[0], 0, atol=1e-6), "NormalizeData failed!")

    def test_encode_categorical_data(self):
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'age': [25, 30, 35]
        })
        preprocessor = EncodeCategoricalData(columns=['gender'])
        df_transformed = preprocessor.process(df)
        self.assertIn('gender_Male', df_transformed.columns, "EncodeCategoricalData failed!")

    def test_outlier_handling_iqr(self):
        df = pd.DataFrame({'col1': [1, 2, 3, 100]})
        preprocessor = HandleOutliers(method='iqr')
        df_transformed = preprocessor.process(df)
        self.assertTrue(df_transformed['col1'].max() <= 99, "HandleOutliers with IQR failed!")
        self.assertTrue(df_transformed['col1'].min() >= -0.75, "HandleOutliers with IQR failed!")

    def test_pipeline(self):
        df = pd.DataFrame({
            'gender': ['Male', 'Female', np.nan],
            'age': [25, np.nan, 35],
            'income': [50000, 60000, np.nan]
        })
        pipeline = PreprocessingPipeline(preprocessors=[
            HandleMissingValues(strategy='mean'),
            NormalizeData(method='minmax'),
            EncodeCategoricalData()
        ])
        df_transformed = pipeline.apply(df)
        self.assertFalse(df_transformed.isnull().any().any(), "Pipeline processing failed!")
        self.assertIn('gender_Male', df_transformed.columns, "Pipeline categorical encoding failed!")


if __name__ == '__main__':
    # This prevents unittest from trying to parse the command line arguments in Jupyter
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
