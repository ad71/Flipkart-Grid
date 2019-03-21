from keras.losses import mean_absolute_error
from scaled_loss import scaled_loss

def composite_loss(y, y_hat):
	'''
	from composite_loss import composite_loss

	model.compile(..., loss=composite_loss)
	'''
    return 0.5 * mean_absolute_error(y, y_hat) + scaled_loss(y, y_hat)