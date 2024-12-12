# test_data_loader.py

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src_telco_churn.data_loader import DataLoader, CSVDataLoader, DataPreparer

class TestDataLoader(unittest.TestCase):

    def test_csv_data_loader_load_data(self):
        # Prepare mock data for testing
        mock_data = {
            'gender': ['Female', 'Male'],
            'tenure': [12, 15],
            'MonthlyCharges': [70.5, 80.0],
            'TotalCharges': [850.0, 1200.0],
            'Churn': ['Yes', 'No']
        }
        mock_df = pd.DataFrame(mock_data)

        # Patch pd.read_csv to return mock data
        with patch('pandas.read_csv', return_value=mock_df):
            loader = CSVDataLoader(required_columns=['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'])
            df = loader.load_data('fake_path.csv')
            # Check if data is loaded correctly
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(df.shape[0], 2)  # Check if two rows are loaded

    def test_csv_data_loader_validate_data_valid(self):
        # Prepare valid data for validation
        valid_data = {
            'gender': ['Female', 'Male'],
            'tenure': [12, 15],
            'MonthlyCharges': [70.5, 80.0],
            'TotalCharges': [850.0, 1200.0],
            'Churn': ['Yes', 'No']
        }
        df = pd.DataFrame(valid_data)

        loader = CSVDataLoader(required_columns=['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'])
        try:
            loader.validate_data(df)
        except ValueError as e:
            self.fail(f"validate_data raised ValueError unexpectedly: {e}")

    def test_csv_data_loader_validate_data_missing_column(self):
        # Prepare data with missing required column
        invalid_data = {
            'gender': ['Female', 'Male'],
            'tenure': [12, 15],
            'MonthlyCharges': [70.5, 80.0],
            'Churn': ['Yes', 'No']
        }
        df = pd.DataFrame(invalid_data)

        loader = CSVDataLoader(required_columns=['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'])
        
        with self.assertRaises(ValueError):
            loader.validate_data(df)

    def test_data_preparer_load_and_validate(self):
        # Prepare mock data for testing
        mock_data = {
            'gender': ['Female', 'Male'],
            'tenure': [12, 15],
            'MonthlyCharges': [70.5, 80.0],
            'TotalCharges': [850.0, 1200.0],
            'Churn': ['Yes', 'No']
        }
        mock_df = pd.DataFrame(mock_data)

        # Patch CSVDataLoader to return mock data
        with patch.object(CSVDataLoader, 'load_data', return_value=mock_df):
            preparer = DataPreparer(loaders=[CSVDataLoader()])
            df = preparer.load_and_validate('fake_path.csv', loader_type='csv')
            # Check if data is validated correctly
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(df.shape[0], 2)  # Check if two rows are loaded

    def test_data_preparer_loader_not_found(self):
        # Test with an unsupported loader type
        preparer = DataPreparer(loaders=[CSVDataLoader()])
        with self.assertRaises(ValueError):
            preparer.load_and_validate('fake_path.csv', loader_type='unsupported')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
