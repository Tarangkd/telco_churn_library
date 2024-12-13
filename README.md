# Telco Churn Prediction Library - Final Project (Computing for Data Science)
- By Tarang Kadyan, Enzo Infantes, Deepak Malik

A Python library for predicting customer churn in telecommunication industry using machine learning. This project is part of the final project for the Computing for Data Science class at Barcelona School of Economics.

# 📂 Repository Contents

- The **library skeleton**, is designed to define the structure for organizing files and folders in a scalable and structured way.

- The **first end-to-end prototype**, which:
  - Loads the dataset.
  - Preprocesses the data.
  - Creates features.
  - Trains a model and performs hyperparameter tuning.
  - Evaluates predictions using specified metrics.

## Dataset Description

The dataset used in this project is the **Telco Customer Churn Dataset**, which includes information about telecom company customers. It is used to predict whether a customer will churn (leave the service).

### Dataset Columns and Description

| **Column Name**       | **Description**                                                       |
|------------------------|----------------------------------------------------------------------|
| `customerID`          | Unique identifier for each customer.                                  |
| `gender`              | Customer’s gender (`Male` or `Female`).                               |
| `SeniorCitizen`       | Indicates if the customer is a senior citizen (`0` or `1`).           |
| `Partner`             | Indicates if the customer has a partner (`Yes` or `No`).              |
| `Dependents`          | Indicates if the customer has dependents (`Yes` or `No`).             |
| `tenure`              | Number of months the customer has been with the company.              |
| `PhoneService`        | Indicates if the customer has a phone service (`Yes` or `No`).        |
| `MultipleLines`       | Indicates if the customer has multiple phone lines (`Yes`, `No`).     |
| `InternetService`     | Type of internet service (`DSL`, `Fiber optic`, `No`).                |
| `OnlineSecurity`      | Indicates if the customer has online security (`Yes` or `No`).        |
| `OnlineBackup`        | Indicates if the customer has online backup (`Yes` or `No`).          |
| `DeviceProtection`    | Indicates if the customer has device protection (`Yes` or `No`).      |
| `TechSupport`         | Indicates if the customer has technical support (`Yes` or `No`).      |
| `StreamingTV`         | Indicates if the customer has streaming TV service (`Yes` or `No`).   |
| `StreamingMovies`     | Indicates if the customer has streaming movie service (`Yes` or `No`).|
| `Contract`            | Type of customer contract (`Month-to-month`, `One year`, `Two year`). |
| `PaperlessBilling`    | Indicates if the customer has paperless billing (`Yes` or `No`).      |
| `PaymentMethod`       | Payment method used (`Electronic check`, `Mailed check`, `Bank transfer`    `Credit card`).                                                                                 |
| `MonthlyCharges`      | Monthly amount charged to the customer.                               |
| `TotalCharges`        | Total amount charged to the customer.                                 |
| `Churn`               | Target variable indicating if the customer churned (`Yes` or `No`).   |


### Dataset Source

The dataset can be downloaded using the Kaggle API:
```python
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", path)

Project Structure

📦 telco-churn-library
┣ 📜 LICENSE                    # Licensing information
┣ 📜 README.md                  # Project documentation
┣ 📜 setup.py                   # Installation script
┣ 📜 requirements.txt           # Dependencies list (currently blank)
┣ 📂 telco_churn                # Main library folder
┃ ┣ 📜 __init__.py              # Package initializer for telco_churn
┃ ┣ 📂 data_processing          # Data processing folder
┃ ┃ ┣ 📜 __init__.py            # Package initializer
┃ ┃ ┣ 📜 data_loader.py         # Data ingestion module
┃ ┃ ┣ 📜 preprocessor.py        # Data transformation module
┃ ┣ 📂 feature_engineering      # Feature engineering folder
┃ ┃ ┣ 📜 __init__.py            # Package initializer
┃ ┃ ┣ 📜 feature_engineering.py # Feature engineering module
┃ ┣ 📂 modeling                 # Modeling folder
┃ ┃ ┣ 📜 __init__.py            # Package initializer
┃ ┃ ┣ 📜 model.py               # Model training and evaluation module
┃ ┣ 📂 api                      # API folder
┃ ┃ ┣ 📜 __init__.py            # Package initializer
┃ ┃ ┣ 📜 api.py                 # API configuration
┃ ┃ ┣ 📜 json_file.json         # JSON data
┃ ┃ ┗ 📜 post_request.py        # API routes and endpoints
┃ ┣ 📂 pipeline                 # End-to-end pipelines
┃ ┃ ┣ 📜 __init__.py            # Package initializer
┃ ┃ ┣ 📜 trained_model.pkl      # The best model
┃ ┃ ┗ 📜 end_to_end.ipynb       # Notebook with all the process of modelling
┃ ┗ 📂 utils                    # Utilities folder
┃   ┣ 📜 __init__.py            # Package initializer
┃   ┗ 📜 utils.py               # General utility functions
┗ 📂 tests                      # Unit tests
  ┣ 📜 __init__.py              # Package initializer for tests
  ┣ 📂 data_processing          # Tests for data processing
  ┃ ┣ 📜 test_data_loader.py
  ┃ ┗ 📜 test_preprocessor.py
  ┣ 📂 feature_engineering      # Tests for feature engineering
  ┃ ┗ 📜 test_feature_engineering.py
  ┣ 📂 modeling                 # Tests for modeling
  ┃ ┗ 📜 test_modeling.py
  ┗ 📂 utils                    # Tests for utilities
    ┗ 📜 test_general.py
```
## Contributors
•	Deepak Malik
-Email: deepak.malik@bse.eu
•	Tarang Kadyan
-Email: tarang.kadyan@bse.eu
•	Enzo Infantes
-Email: enzo.infantes@bse.eu


## GENERAL GUIDELINES 

To scale a machine learning library and enable the easy addition of new preprocessors, features, models, and metrics, one should follow a modular, maintainable, and extensible design pattern. Here are some guidelines based on the work we've done with the current project, adhering to best practices like DRY (Don't Repeat Yourself), object-oriented programming (OOP), error handling, unit testing, loose coupling, and more.

## 1. Follow the SOLID Principles
S: Single Responsibility Principle
Each class, function, or module should have one responsibility. For example, your DataPreparer class should focus solely on data preparation tasks, and not mix in model training or evaluation tasks.

O: Open/Closed Principle
The system should be open for extension but closed for modification. This allows you to add new preprocessors, models, and features without altering existing code. For instance:
Create abstract base classes (ABC) for Preprocessor, Model, Feature, and Metric that can be easily extended by adding new subclasses.
Use polymorphism to allow different model types (e.g., LogisticRegressionModel, RandomForestModel) to be handled interchangeably.

L: Liskov Substitution Principle
Ensure that derived classes can be used in place of their base class without altering the expected behavior. For example, any new Preprocessor should conform to the base Preprocessor class interface, so it can be added seamlessly to the pipeline.

I: Interface Segregation Principle
Avoid forcing clients to depend on interfaces they don’t use. For example, don’t combine unrelated functionalities into a single class or interface.

D: Dependency Inversion Principle
Depend on abstractions, not concrete implementations. This is crucial for loose coupling. For example, your model training code should not directly reference a specific model like LogisticRegressionModel but should depend on an abstract Model class.

## 2. Modular Code Structure
Organize your library into separate modules for each type of functionality, such as:

- preprocessing.py: Contains various preprocessing techniques (e.g., handling missing values, normalizing data, encoding categorical features).
- feature_engineering.py: Contains the feature engineering methods (e.g., statistical features, interaction features, derived features).
- modeling.py: Defines model training classes and methods for different models, hyperparameter tuning, and pipelines.
- metrics.py: Provides functions to evaluate model performance, such as accuracy, precision, recall, ROC-AUC, etc.

Each of these modules can then be imported into a pipeline.py or main.py file, where they are composed into a full machine learning pipeline

## 3. Use Classes for Encapsulation and Reusability
In Python, classes are great for encapsulating related methods and attributes. This will help keep the code clean and reusable.

- Preprocessing: Create a PreprocessingPipeline class that accepts a list of individual preprocessing steps as arguments. Each preprocessor should be an instance of a class with a common interface (fit, transform).
- Feature Engineering: Similar to preprocessing, create a FeatureEngineeringPipeline class that applies multiple feature engineering techniques.
- Modeling: Each model (e.g., LogisticRegressionModel) should inherit from an abstract Model class with methods like fit, predict, and evaluate. This ensures that adding new models is straightforward.

## 4. Keep Code DRY (Don't Repeat Yourself)
Minimize code duplication by abstracting common functionality into reusable functions or classes. For example, the logic for handling missing values, scaling, and encoding should be encapsulated in individual preprocessors, which can be reused across multiple datasets and pipelines.

## 5. Error Handling and Logging
Use proper error handling and logging throughout your code to make it more robust and easier to debug.

Error Handling: Handle specific exceptions using try-except blocks and provide meaningful error messages.
Logging: Use Python’s logging module to track the execution of code, which is especially useful when scaling.

## 6. Loose Coupling
Loose coupling refers to designing modules or classes in such a way that they are independent of each other. This increases flexibility and maintainability.

Decouple Data Loading: The DataLoader class should not depend on a specific model or preprocessor. It should focus on reading the data and passing it to other modules.
Decouple Model and Evaluation: The ModelEvaluator should be agnostic of the specific model being evaluated. It should be able to evaluate any model class that implements the predict method.

## 7. Unit Testing
For scalability, we need comprehensive unit tests. These tests should focus on individual functions and classes to ensure correctness, stability, and performance.

- Test preprocessors: Ensure that preprocessing steps like handling missing values, encoding, and scaling work as expected.
- Test feature engineering: Verify that feature engineering steps generate the expected results, especially for complex transformations.
- Test models: Ensure models are correctly trained and predictions are made with the correct output shape and expected values.

## 8. Scalability Considerations
As we add more functionality to the library (new models, preprocessors, features, etc.), we need to ensure that:

- The system handles large datasets efficiently by optimizing memory usage and processing time. This might involve using sparse matrices or more efficient data structures.
- The code is easy to extend by following object-oriented principles like inheritance and composition. New models and transformers should be added without modifying the existing codebase.
- Documentation is provided for every module, class, and function to ensure others can easily understand and extend your code.

## 9. Adding New Features, Models, and Metrics
To extend your library in the future:
- Preprocessors: Create new preprocessors by extending the base Preprocessor class and implementing the transform method.
- Feature Engineering: Implement new feature engineering methods as separate classes and integrate them into your FeaturePipeline.
- Models: Add new models by inheriting from the Model class and implementing the required methods (fit, predict, evaluate).
- Metrics: Add custom metrics by creating a new function or class, ensuring they conform to a consistent API.

## 10. Continuous Integration (CI) and Documentation
As your library grows, integrate CI tools like GitHub Actions or Travis CI to automatically run your tests when code is pushed to the repository. Also, consider generating documentation with tools like Sphinx to ensure your code is well-documented.

## Conclusion:
By following these guidelines, one can scale their library efficiently, adding new functionality without making the existing codebase fragile. Focus on maintaining a modular, extensible, and testable code structure, adhering to object-oriented principles, and ensuring proper error handling and logging.
