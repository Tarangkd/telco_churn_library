import pandas as pd
from abc import ABC, abstractmethod

# Abstract Base Class for Data Loader
class DataLoader(ABC):
    """
    Abstract Base Class for defining data loaders.
    Enforces implementation of load_data and validate_data methods.
    """

    @abstractmethod
    def load_data(self, file_path: str):
        pass

    @abstractmethod
    def validate_data(self, df: pd.DataFrame):
        pass

class CSVDataLoader(DataLoader):
    def __init__(self, required_columns=None):
        """
        Initialize the CSV Data Loader with optional required columns.
        Args:
            required_columns (list): List of columns required for the pipeline.
        """
        self.required_columns = required_columns or []

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        Args:
            file_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def validate_data(self, df: pd.DataFrame, column_to_convert: str = None):
        """
        Validate the loaded data and optionally transform a specified column.
        Args:
            df (pd.DataFrame): The DataFrame to validate.
            column_to_convert (str): Name of the column to convert to numeric and replace blanks with NaN.
        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        # Check for missing required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if the dataset is empty or contains only NaN values
        if df.isnull().all().all():
            raise ValueError("The dataset appears to be empty or contains only NaN values.")

        # Transform the specified column if provided
        if column_to_convert:
            if column_to_convert in df.columns:
                df[column_to_convert] = df[column_to_convert].replace(' ', pd.NA)  # Replace blanks with NaN
                df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors='coerce')  # Convert to numeric
                print(f"Column '{column_to_convert}' successfully converted to numeric and blanks replaced with NaN.")
            else:
                raise ValueError(f"Column '{column_to_convert}' not found in the dataset.")

        print("Data validation successful.")

class DataPreparer:
    def __init__(self, loaders: list):
        """
        Initialize with a list of data loaders.
        Args:
            loaders (list): List of DataLoader instances.
        """
        self.loaders = loaders

    def load_and_validate(self, file_path: str, loader_type: str, column_to_convert: str = None) -> pd.DataFrame:
        """
        Load and validate data using the appropriate loader.
        Args:
            file_path (str): Path to the data file.
            loader_type (str): Type of loader (e.g., 'csv').
            column_to_convert (str): Name of the column to convert to numeric and replace blanks with NaN.
        Returns:
            pd.DataFrame: Validated DataFrame.
        """
        # Modified to use 'csv' instead of 'csvdataloader'
        loader_map = {loader.__class__.__name__.replace('DataLoader', '').lower(): loader for loader in self.loaders}

        if loader_type not in loader_map:
            raise ValueError(f"Loader type '{loader_type}' is not supported. Available types: {list(loader_map.keys())}")

        loader = loader_map[loader_type]
        df = loader.load_data(file_path)
        loader.validate_data(df, column_to_convert=column_to_convert)
        return df
