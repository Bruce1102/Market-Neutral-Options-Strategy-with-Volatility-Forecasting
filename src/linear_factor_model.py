import numpy as np
from scipy.optimize import minimize
from typing import List, Union

class LinearFactorModel:
    """
    A class to represent a Linear Factor Model for regression.
    
    Attributes:
    params (numpy.ndarray) : The learned parameters of the linear model.
    """

    def __init__(self) -> None:
        """Initialize the LinearFactorModel class."""
        self.params: Union[np.ndarray, None] = None
    
    def _objective_function(self, params: np.ndarray, target: np.ndarray, factors: np.ndarray) -> float:
        """
        Objective function to compute the mean squared error between the 
        predicted values using the linear factor model and the actual values.

        Parameters:
        - params  (numpy.ndarray)  : Parameters of the linear model.
        - target  (numpy.ndarray)  : Actual target values.
        - factors  (numpy.ndarray) : Factor matrix.

        Returns:
        - float : Mean squared error between predicted and actual values.
        """
        # Create an array with the first column as ones (for the intercept) and the subsequent columns as factors
        target_factors = np.column_stack([np.ones(factors.shape[0]), factors])

        # Predict the target using the linear factor model
        predicted_target = np.dot(target_factors, params)
    
        # Compute the mean squared error
        mse = np.mean((predicted_target - target) ** 2)
        return mse

    def fit(self, target: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """
        Fit the Linear Factor Model on given target and factors.

        Parameters:
        ----------
        - target (numpy.ndarray)  : Actual target values.
        - factors (numpy.ndarray) : Factor matrix.

        Returns:
        - np.ndarray: Learned parameters of the model.
        """
        # Initial parameters set to zeros
        initial_params = np.zeros(factors.shape[1] + 1)
        
        # Optimize the objective function
        self.params = minimize(self._objective_function, initial_params, args=(target, factors)).x
        
        return self.params

    def predict(self, factors: np.ndarray) -> np.ndarray:
        """
        Predict values using the trained Linear Factor Model.

        Parameters:
        - factors (numpy.ndarray) : Factor matrix. Can be 2D or 3D for batched input.

        Returns:
        - np.ndarray: Predicted values. 2D or 3D depending on input.
        """
        # Ensure the model is trained
        if self.params is None:
            raise ValueError("Model is not yet trained. Use the 'fit' method first.")

        # Check if the input is batched (3D)
        if len(factors.shape) == 3:
            batched = True
            batch_size, num_samples, num_factors = factors.shape
            factors = factors.reshape(-1, num_factors)
        else:
            batched = False

        # Create an array with the first column as ones (for the intercept) and the subsequent columns as factors
        prediction_factors = np.column_stack([np.ones(factors.shape[0]), factors])

        # Predict the target using the linear factor model
        predictions = np.dot(prediction_factors, self.params)

        # Reshape the predictions back to 3D if the input was batched
        if batched:
            predictions = predictions.reshape(batch_size, num_samples)

        return predictions
