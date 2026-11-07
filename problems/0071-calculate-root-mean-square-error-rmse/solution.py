
import numpy as np

def rmse(y_true, y_pred):
	errors = y_pred.astype(float) - y_true.astype(float)

	rmse = np.sqrt(np.mean(errors**2))

	return round(float(rmse),3)
