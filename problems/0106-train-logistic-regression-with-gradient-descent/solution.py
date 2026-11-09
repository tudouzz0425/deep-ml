import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	n_samples = X.shape[0]
	X_bias = np.column_stack((np.ones(n_samples), X))
	theta = np.zeros(X_bias.shape[1])
	losses = []

	for _ in range(iterations):
		z = X_bias @ theta
		probabilities = np.exp(-np.logaddexp(0.0,-z))

		sample_losses = np.logaddexp(0.0, np.where(y == 1, -z, z))
		losses.append(float(sample_losses.sum()))

		gradient = X_bias.T @ (probabilities - y)
		theta -= learning_rate * gradient

	coefficients = [round(value, 4) for value in theta]
	return coefficients,[round(loss, 4) for loss in losses]