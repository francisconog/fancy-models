# BaseEstimator

The `BaseEstimator` class is a foundational class for all estimators in the scikit-learn library. It provides a standard interface and common functionality that all estimators (such as classifiers, regressors, and transformers) should implement. Here's a concise breakdown of its purpose, functionality, and key concepts:

## Purpose

The BaseEstimator serves as a base class for creating custom machine learning models in scikit-learn. It ensures that all estimators have a consistent API, which is crucial for the interoperability of different components within the library.

## Key Functionalities

1. Parameter Management:

    - `get_params(self, deep=True)`: This method retrieves the parameters of the estimator. It returns a dictionary mapping parameter names to their values. The deep parameter determines whether to include parameters of nested estimators.

    - `set_params(self, **params)`: Sets the parameters of the estimator. This is useful for hyperparameter tuning and model configuration.

2. Cloning:

    - `clone(estimator, safe=True)`: Although not a method of BaseEstimator, this function is crucial for creating a new instance of an estimator with the same parameters but without any fitted data. This is essential for model evaluation and cross-validation.

3. Representation:

    - `__repr__(self, N_CHAR_MAX=700)`: Provides a string representation of the estimator, which is useful for debugging and logging. It ensures that the estimator's parameters are displayed in a readable format.

4. Validation:

    - `_validate_data(self, X, y, reset=True, validate_separately=False, **check_params)`: Validates input data and sets or checks the `n_features_in_` attribute, ensuring that the input data is consistent with what the estimator expects.

5. Feature Names Handling:

    - `_check_feature_names(self, X, reset)`: Manages feature names, ensuring that they are consistent across different datasets. This is important for models that rely on specific feature names.

6. Serialization:

    - `__getstate__(self)` and `__setstate__(self, state)`: Handle the serialization and deserialization of the estimator, which is important for model persistence.

## Concepts

- **Introspection**: The class uses Python's introspection capabilities (via `inspect`) to dynamically retrieve parameter names and manage them.

- **Mixin Pattern**: The code includes several mixin classes (e.g., `ClassifierMixin`, `RegressorMixin`) that provide additional functionality specific to certain types of estimators.

- **Tags**: The `_get_tags()` method provides a mechanism for defining and retrieving metadata about the estimator, which can influence its behavior in certain contexts.

## Implementing Custom Models

To implement a custom machine learning model using `BaseEstimator`, you would typically:

1. Subclass `BaseEstimator` and any relevant mixins (e.g., `ClassifierMixin` for classifiers).

2. Define an `__init__` method that explicitly lists all hyperparameters.

3. Implement the `fit()` method to train the model.

4. Implement the `predict()` method (and `score()` if applicable) to make predictions.

5. Optionally, override `_validate_data()` to customize input validation.

By adhering to this structure, your custom model will integrate seamlessly with scikit-learn's ecosystem, benefiting from its tools for model selection, evaluation, and deployment.

## Exemples

```python
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np

# Custom Classifier Example
class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        # Simple fit method that calculates the mean of each feature for each class
        self.classes_ = np.unique(y)
        self.means_ = {cls: X[y == cls].mean(axis=0) for cls in self.classes_}
        return self

    def predict(self, X):
        # Predict based on the closest mean
        distances = {cls: np.linalg.norm(X - mean, axis=1) for cls, mean in self.means_.items()}
        return np.array([self.classes_[np.argmin([dist[i] for dist in distances.values()])] for i in range(X.shape[0])])

# Custom Regressor Example
class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, factor=1.0):
        self.factor = factor

    def fit(self, X, y):
        # Simple fit method that calculates the mean of the target
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        # Predict by scaling the mean with the factor
        return np.full(X.shape[0], self.mean_ * self.factor)

# Example usage
if __name__ == "__main__":
    # Toy data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_class = np.array([0, 0, 1, 1])
    y_reg = np.array([1.0, 2.0, 3.0, 4.0])

    # Custom Classifier
    clf = CustomClassifier(threshold=0.5)
    clf.fit(X, y_class)
    print("Classifier Predictions:", clf.predict(X))

    # Custom Regressor
    reg = CustomRegressor(factor=1.5)
    reg.fit(X, y_reg)
    print("Regressor Predictions:", reg.predict(X))
```