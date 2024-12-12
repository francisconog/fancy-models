# Solver

The [Solver class](https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/solver.py) is designed for data imputation, which is the process of filling in missing values in a dataset. Here's a breakdown of its main components and how you can extend it:

## Main Purpose

The Solver class provides a framework for handling missing data in a 2D NumPy array. It offers several methods to fill missing values using different strategies and ensures that the data is prepared and validated before imputation.

## Key Methods

1. `__init__`: Initializes the solver with parameters for the fill method, value constraints, and a normalizer.

2. `fill`: Fills missing values in the data array `X` based on the specified fill_method. Supported methods include:

    - `"zero"`: Fill with zeros.
    - `"mean"`: Fill with column means.
    - `"median"`: Fill with column medians.
    - `"min"`: Fill with the minimum value per column.
    - `"random"`: Fill with random samples based on column statistics.

3. `prepare_input_data`: Validates the input data and creates a mask indicating missing values.

4. `clip`: Ensures that the values in X fall within specified minimum and maximum bounds.

5. `project_result`: Applies inverse normalization and clipping to the imputed data.

6. `solve`: A placeholder method meant to be overridden in subclasses. It should implement the specific imputation algorithm. This is the primary method you must override is the solve method.

7. `fit_transform`: Combines data preparation, filling, solving, and post-processing to return a complete dataset.

8. `fit`: Fits the imputer on input `X`. The `fit` method is not explicitly mentioned as a method that needs to be overridden for a custom solver. However, if your custom imputation algorithm requires specific fitting logic (e.g., learning parameters from the data), you might consider overriding `fit` as well. But typically, the focus is on the `solve` method for implementing the core imputation strategy.

9. `transform`: Applies the learned parameters (from fit) to the data to perform the actual transformation.


## Extending the Solver

To implement a custom algorithm for data imputation, you should create a subclass of Solver and override the solve method. Here's a basic template:

```python
class CustomSolver(Solver):
    def solve(self, X, missing_mask):
        # Implement your custom imputation logic here
        # For example, a simple fill with a constant value
        X[missing_mask] = 42  # Replace 42 with your logic
        return X

```

## Concepts Used

- **Data** Imputation: The process of replacing missing data with substituted values.
- **NumPy**: A library for numerical operations on arrays.
- **Normalization**: Adjusting values measured on different scales to a common scale.
- **Masking**: Using boolean arrays to identify missing data.
- **Inheritance**: Extending the functionality of a base class by creating a subclass.

By extending the `Solver` class, you can implement custom imputation strategies tailored to specific datasets or requirements.

