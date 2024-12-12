import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from typing import List


# Abstract Base Class for Preprocessors
class Preprocessor(ABC):
    """
    Abstract Base Class for creating preprocessors.
    Ensures all preprocessing classes implement the 'process' method.
    """
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# Preprocessor 1: Handle Missing Values
class HandleMissingValues(Preprocessor):
    def __init__(self, strategy: str = 'mean', columns: List[str] = None):
        """
        Initialize with a strategy for imputation.
        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
            columns (List[str]): Specific columns to handle. If None, applies to all.
        """
        self.strategy = strategy
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            imputer = SimpleImputer(strategy=self.strategy)
            columns_to_impute = self.columns or df.select_dtypes(include=['number']).columns
            df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
            return df
        except Exception as e:
            raise ValueError(f"Error in HandleMissingValues: {e}")


# Preprocessor 2: Normalize Numeric Columns
class NormalizeData(Preprocessor):
    def __init__(self, method: str = 'minmax', columns: List[str] = None):
        """
        Initialize with a normalization method.
        Args:
            method (str): Normalization method ('minmax' or 'zscore').
            columns (List[str]): Specific columns to normalize. If None, applies to all numeric columns.
        """
        self.method = method
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            scaler = MinMaxScaler() if self.method == 'minmax' else StandardScaler()
            columns_to_scale = self.columns or df.select_dtypes(include=['number']).columns
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            return df
        except Exception as e:
            raise ValueError(f"Error in NormalizeData: {e}")


# Preprocessor 3: Encode Categorical Data
class EncodeCategoricalData(Preprocessor):
    def __init__(self, columns: List[str] = None, drop_first: bool = False):
        """
        Initialize with an option to one-hot encode.
        Args:
            columns (List[str]): Specific columns to encode. If None, applies to all categorical columns.
            drop_first (bool): Whether to drop the first column for k-1 encoding.
        """
        self.columns = columns
        self.drop_first = drop_first

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            categorical_cols = self.columns or df.select_dtypes(include=['object', 'category']).columns
            encoder = OneHotEncoder(drop='first' if self.drop_first else None, sparse_output=False)  # Fixed here
            for col in categorical_cols:
                one_hot_encoded = encoder.fit_transform(df[[col]])
                one_hot_df = pd.DataFrame(
                    one_hot_encoded, 
                    columns=[f"{col}_{category}" for category in encoder.categories_[0]]
                )
                df = pd.concat([df.drop(col, axis=1), one_hot_df], axis=1)
            return df
        except Exception as e:
            raise ValueError(f"Error in EncodeCategoricalData: {e}")



# Preprocessor 4: Handle Outliers
class HandleOutliers(Preprocessor):
    def __init__(self, method: str = 'iqr', columns: List[str] = None):
        """
        Initialize with an outlier handling method.
        Args:
            method (str): Outlier detection method ('iqr' or 'zscore').
            columns (List[str]): Specific columns to handle. If None, applies to all numeric columns.
        """
        self.method = method
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            columns_to_handle = self.columns or df.select_dtypes(include=['number']).columns
            for col in columns_to_handle:
                if self.method == 'iqr':
                    q1, q3 = np.percentile(df[col], [25, 75])
                    iqr = q3 - q1
                    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
                elif self.method == 'zscore':
                    z_scores = (df[col] - df[col].mean()) / df[col].std()
                    df[col] = np.where(np.abs(z_scores) > 3, np.sign(z_scores) * 3 * df[col].std(), df[col])
            return df
        except Exception as e:
            raise ValueError(f"Error in HandleOutliers: {e}")


# Universal Preprocessing Pipeline
class PreprocessingPipeline:
    def __init__(self, preprocessors: List[Preprocessor]):
        """
        Initialize the pipeline with a list of preprocessors.
        Args:
            preprocessors (List[Preprocessor]): A list of preprocessors to apply.
        """
        self.preprocessors = preprocessors

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for preprocessor in self.preprocessors:
                df = preprocessor.process(df)
            return df
        except Exception as e:
            raise ValueError(f"Error in PreprocessingPipeline: {e}")


# Example Unit Tests for Validation
class TestPreprocessingPipeline:
    def test_handle_missing_values(self):
        df = pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, 6]})
        preprocessor = HandleMissingValues(strategy='mean')
        df_transformed = preprocessor.process(df)
        assert not df_transformed.isnull().any().any(), "HandleMissingValues failed!"

    def test_normalize_data(self):
        df = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessor = NormalizeData(method='zscore')
        df_transformed = preprocessor.process(df)
        assert np.isclose(df_transformed.mean().values[0], 0, atol=1e-6), "NormalizeData failed!"

##Example usage
if __name__ == "__main__":
    # Load a sample dataset
    df = pd.DataFrame({
        'gender': ['Male', 'Female', np.nan],
        'age': [25, np.nan, 35],
        'income': [50000, 60000, np.nan]
    })

    # Instantiate the pipeline
    pipeline = PreprocessingPipeline(preprocessors=[
        HandleMissingValues(strategy='mean'),
        NormalizeData(method='minmax'),
        EncodeCategoricalData()
    ])

    # Apply preprocessing
    df_transformed = pipeline.apply(df)
    print(df_transformed.head())