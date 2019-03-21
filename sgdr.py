from keras.callbacks import Callback
import keras.backend as K
import numpy as np

class SGDRScheduler(Callback):
	''' Learning rate scheduler with cyclic restarts

	Usage:
	sgdr = SGDRScheduler(minimum=1e-5
						 maximum=1e-2,
						 step_size=(train_generator.n//train_generator.batch_size),
						 lr_decay=0.9,
						 cycle_len=1,
						 cycle_mult=2)
	model.fit(x_train, y_train, epochs=100, callbacks=[sgdr])
	'''

	def __init__(self, minimum, maximum, step_size, lr_decay=0.9, cycle_len=1, cycle_mult=2):
		self.minimum = minimum
		self.maximum = maximum
		self.lr_decay = lr_decay

		self.prev_restart = 0
		self.next_restart = cycle_len

		self.step_size = step_size

		self.cycle_len = cycle_len
		self.cycle_mult = cycle_mult
		self.history = {}

	def clr(self):
		''' Calculate the learning rate by cosine annealing '''
		fraction_to_restart = self.prev_restart / (self.step_size * self.cycle_len)
		lr = self.minimum + 0.5 * (self.maximum - self.minimum) * (1 + np.cos(fraction_to_restart * np.pi))
		return lr

	def on_train_begin(self, logs={}):
		''' Initialize the learning rate to the minimum value at the start of training '''
		logs = logs or {}
		K.set_value(self.model.optimizer.lr, self.maximum)

	def on_batch_end(self, batch, logs={}):
		''' Record the previous batch statistics and update the learning rate '''
		logs = logs or {}
		self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

		self.prev_restart += 1
		K.set_value(self.model.optimizer.lr, self.clr())

	def on_epoch_end(self, epoch, logs={}):
		''' Check for end of current cycle, apply restarts when necessary '''
		if epoch + 1 == self.next_restart:
			self.prev_restart = 0
			self.cycle_len = np.ceil(self.cycle_len * self.cycle_mult)
			self.next_restart += self.cycle_len
			self.maximum *= self.lr_decay
			self.best_weights = self.model.get_weights()

	def on_train_end(self, logs={}):
		''' Set weights to the values from the end of the most recent cycle for best performance '''
		self.model.set_weights(self.best_weights)
