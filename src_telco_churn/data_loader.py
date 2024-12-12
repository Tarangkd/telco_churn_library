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


# Implementation: CSV Data Loader
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

    def validate_data(self, df: pd.DataFrame):
        """
        Validate the loaded data.
        Args:
            df (pd.DataFrame): The DataFrame to validate.
        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if df.isnull().all().all():
            raise ValueError("The dataset appears to be empty or contains only NaN values.")
        print("Data validation successful.")


# Implementation: Universal Data Preparer
class DataPreparer:
    def __init__(self, loaders: list):
        """
        Initialize with a list of data loaders.
        Args:
            loaders (list): List of DataLoader instances.
        """
        self.loaders = loaders

    def load_and_validate(self, file_path: str, loader_type: str) -> pd.DataFrame:
        """
        Load and validate data using the appropriate loader.
        Args:
            file_path (str): Path to the data file.
            loader_type (str): Type of loader (e.g., 'csv').
        Returns:
            pd.DataFrame: Validated DataFrame.
        """
        # Modified to use 'csv' instead of 'csvdataloader'
        loader_map = {loader.__class__.__name__.replace('DataLoader', '').lower(): loader for loader in self.loaders}

        if loader_type not in loader_map:
            raise ValueError(f"Loader type '{loader_type}' is not supported. Available types: {list(loader_map.keys())}")

        loader = loader_map[loader_type]
        df = loader.load_data(file_path)
        loader.validate_data(df)
        return df

# Example Usage
if __name__ == "__main__":
    # Define required columns for your pipeline
    required_columns = ['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']

    # Initialize CSV Data Loader
    csv_loader = CSVDataLoader(required_columns=required_columns)

    # Initialize Data Preparer with loaders
    preparer = DataPreparer(loaders=[csv_loader])

    # Load and validate data
    file_path = "/Users/tarangkadyan/Downloads/telco_churn_library/Data/data.csv"  # Update with actual file path
    data = preparer.load_and_validate(file_path, loader_type="csv")

    print(data.head())


