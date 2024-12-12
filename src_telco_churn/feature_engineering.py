import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List


# Abstract Base Class for Feature Engineering
class FeatureEngineer(ABC):
    """
    Abstract Base Class for creating feature transformers.
    Ensures all feature engineering classes implement the 'transform' method.
    """
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# Helper Functions for Reusability
def detect_columns_by_dtype(df: pd.DataFrame, include_types: List[str]) -> List[str]:
    """
    Detect columns in a DataFrame by their data types.
    Args:
        df (pd.DataFrame): Input DataFrame.
        include_types (List[str]): List of data types to include (e.g., 'object', 'number').
    Returns:
        List[str]: List of column names matching the specified types.
    """
    return df.select_dtypes(include=include_types).columns.tolist()


# Feature 1: Statistical Aggregations for Numeric Columns
class StatisticalFeatures(FeatureEngineer):
    def __init__(self, group_by_column: str = None):
        """
        Initialize with an optional group-by column for aggregation.
        Args:
            group_by_column (str): Column to group by. If None, applies globally.
        """
        self.group_by_column = group_by_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols = detect_columns_by_dtype(df, include_types=['number'])
            if self.group_by_column and self.group_by_column in df.columns:
                stats = df.groupby(self.group_by_column)[numeric_cols].agg(['mean', 'std', 'max', 'min'])
                stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
                return df.merge(stats, on=self.group_by_column, how='left')
            else:
                for col in numeric_cols:
                    df[f'{col}_mean'] = df[col].mean()
                    df[f'{col}_std'] = df[col].std()
                return df
        except Exception as e:
            raise ValueError(f"Error in StatisticalFeatures: {e}")


# Feature 2: Encoding Categorical Columns
class CategoricalEncoding(FeatureEngineer):
    def __init__(self):
        self.encoders = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            categorical_cols = detect_columns_by_dtype(df, include_types=['object', 'category'])
            for col in categorical_cols:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            return df
        except Exception as e:
            raise ValueError(f"Error in CategoricalEncoding: {e}")


# Feature 3: Interaction Features
class InteractionFeatures(FeatureEngineer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols = detect_columns_by_dtype(df, include_types=['number'])
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            return df
        except Exception as e:
            raise ValueError(f"Error in InteractionFeatures: {e}")


# Feature 4: Temporal Features
class TemporalFeatures(FeatureEngineer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            datetime_cols = detect_columns_by_dtype(df, include_types=['datetime'])
            for col in datetime_cols:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.weekday
            return df
        except Exception as e:
            raise ValueError(f"Error in TemporalFeatures: {e}")


# Feature 5: Derived Features
class DerivedFeatures(FeatureEngineer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
            if 'SeniorCitizen' in df.columns:
                df['is_senior'] = (df['SeniorCitizen'] == 1).astype(int)
            return df
        except Exception as e:
            raise ValueError(f"Error in DerivedFeatures: {e}")


# Universal Feature Engineering Pipeline
class FeaturePipeline:
    def __init__(self, transformers: List[FeatureEngineer]):
        """
        Initialize the pipeline with a list of transformers.
        Args:
            transformers (List[FeatureEngineer]): A list of feature transformers.
        """
        self.transformers = transformers

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for transformer in self.transformers:
                df = transformer.transform(df)
            return df
        except Exception as e:
            raise ValueError(f"Error in FeaturePipeline: {e}")


# Example Unit Tests for Loose Coupling and Validation
class TestFeaturePipeline:
    def test_statistical_features(self):
        df = pd.DataFrame({'group': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40]})
        transformer = StatisticalFeatures(group_by_column='group')
        df_transformed = transformer.transform(df)
        assert 'value_mean' in df_transformed.columns, "StatisticalFeatures failed!"

    def test_categorical_encoding(self):
        df = pd.DataFrame({'category': ['a', 'b', 'a']})
        transformer = CategoricalEncoding()
        df_transformed = transformer.transform(df)
        assert 'category' in df_transformed.columns, "CategoricalEncoding failed!"

