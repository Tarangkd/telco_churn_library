import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings("ignore")

# Abstract Base Class for Models
class Model(ABC):
    """
    Abstract Base Class for defining models.
    Enforces implementation of train, predict, and evaluate methods.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        """
        Initialize the model with hyperparameters.
        Args:
            kwargs: Hyperparameters for the LogisticRegression model.
        """
        self.model = LogisticRegression(**kwargs)

    def train(self, X_train, y_train):
        """
        Train the model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on test data.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, y_pred):
        y_pred_proba = self.model.predict_proba(X_test)
        
        if len(np.unique(y_test)) > 2:
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            }
        else:
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) 
            }



# Task 1: Train-Test Split
class TrainTestSplit:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize with test size and random state.
        Args:
            test_size (float): Proportion of the data to include in the test split.
            random_state (int): Random state for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """
        Split the data into train and test sets.
        """
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)


# Task 2: Hyperparameter Tuning
class HyperparameterTuner:
    def __init__(self, model: BaseEstimator, param_grid: dict, cv: int = 5, scoring: str = 'accuracy'):
        """
        Initialize with model, parameter grid, and cross-validation settings.
        Args:
            model (BaseEstimator): The model to tune.
            param_grid (dict): Hyperparameter grid.
            cv (int): Number of folds in cross-validation.
            scoring (str): Scoring metric for evaluation.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def tune(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV.
        """
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_


# Task 3: Cross-Validation
class CrossValidator:
    def __init__(self, model: BaseEstimator, cv: int = 5, scoring: str = 'accuracy'):
        """
        Initialize with model, cross-validation settings, and scoring metric.
        Args:
            model (BaseEstimator): The model to validate.
            cv (int): Number of folds in cross-validation.
            scoring (str): Scoring metric for evaluation.
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring

    def validate(self, X, y):
        """
        Perform cross-validation and return scores.
        """
        scores = cross_val_score(self.model, X, y, cv=self.cv, scoring=self.scoring)
        return scores.mean(), scores.std()


# Task 4: Unified Pipeline for Modeling
class ModelingPipeline:
    def __init__(self, model: Model, tuner: HyperparameterTuner = None):
        """
        Initialize the pipeline with model and optional tuner.
        Args:
            model (Model): The base model to use.
            tuner (HyperparameterTuner): The hyperparameter tuner.
        """
        self.model = model
        self.tuner = tuner

    def run(self, X, y):
        """
        Execute the full modeling pipeline.
        """
        # Split data
        splitter = TrainTestSplit()
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # Hyperparameter tuning (if applicable)
        if self.tuner:
            print("Performing hyperparameter tuning...")
            self.model.model, best_params = self.tuner.tune(X_train, y_train)
            print(f"Best Parameters: {best_params}")

        # Train model
        self.model.train(X_train, y_train)

        # Predict
        y_pred = self.model.predict(X_test)

        # Evaluate model
        metrics = self.model.evaluate(X_test, y_test, y_pred)  
        print("Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        return metrics