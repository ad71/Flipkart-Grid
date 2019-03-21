import numpy as np
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(lr=1e-3, lr_decay=0.75, step_size=10):
	''' Simple step-decay scheduler '''
	def schedule(epoch):
		return lr * (lr_decay ** np.floor(epoch / step_size))

	return LearningRateScheduler(schedule)
