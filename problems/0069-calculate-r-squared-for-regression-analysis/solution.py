
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	y_mean = np.mean(y_true)

	ss_res = np.sum((y_pred - y_true) ** 2)

	ss_tot = np.sum((y_mean - y_true) ** 2)

	r2 = 1 - ss_res/ss_tot
	
	return r2
