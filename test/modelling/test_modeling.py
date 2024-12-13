import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from telco_churn.modelling.model import LogisticRegressionModel, HyperparameterTuner, ModelingPipeline, CrossValidator, TrainTestSplit

class TestModeling(unittest.TestCase):

    # Test LogisticRegressionModel
    def test_logistic_regression_model(self):
        # Prepare test data
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
        
        # Create the model
        model = LogisticRegressionModel(solver='liblinear')
        
        # Train the model
        model.train(X, y)
        
        # Predict with the model
        y_pred = model.predict(X)
        
        # Evaluate the model
        metrics = model.evaluate(X, y, y_pred)
        
        # Check that evaluation metrics are returned
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)

    # Test TrainTestSplit
    def test_train_test_split(self):
        # Prepare test data
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
        
        # Create the TrainTestSplit instance
        splitter = TrainTestSplit(test_size=0.3)
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        
        # Assert that the split works correctly
        self.assertEqual(X_train.shape[0] + X_test.shape[0], X.shape[0])
        self.assertEqual(y_train.shape[0] + y_test.shape[0], y.shape[0])

    # Test HyperparameterTuner
    def test_hyperparameter_tuner(self):
        # Prepare test data
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
        
        # Create the model and tuner
        rf_model = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        tuner = HyperparameterTuner(rf_model, param_grid)
        
        # Perform hyperparameter tuning
        best_model, best_params = tuner.tune(X, y)
        
        # Assert that the best model has been returned
        self.assertIsInstance(best_model, RandomForestClassifier)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)

    # Test CrossValidator
    def test_cross_validation(self):
        # Prepare test data
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
        
        # Create the model and cross-validator
        rf_model = RandomForestClassifier()
        cv_validator = CrossValidator(rf_model, cv=5, scoring='accuracy')
        
        # Perform cross-validation
        mean_score, std_score = cv_validator.validate(X, y)
        
        # Assert that mean and std scores are returned
        self.assertGreaterEqual(mean_score, 0)
        self.assertGreaterEqual(std_score, 0)

    # Test the full ModelingPipeline
    def test_modeling_pipeline(self):
        # Prepare test data
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
        
        # Create the model, tuner, and pipeline
        model = LogisticRegressionModel(solver='liblinear')
        tuner = HyperparameterTuner(RandomForestClassifier(), param_grid={'n_estimators': [50, 100], 'max_depth': [3, 5]})
        pipeline = ModelingPipeline(model=model, tuner=tuner)
        
        # Run the pipeline
        metrics = pipeline.run(X, y)
        
        # Check that metrics are returned
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)