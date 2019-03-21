import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback

class LRFinder(Callback):

	'''
	lrf = LRFinder(minimum=3e-6, maximum=3e-4, step_size=(train_generator.n//train_generator.batch_size), n_epochs=3)
	model.fit(x_train, y_train, callbacks=[lrf, ])

	lrf.lr_graph()
	lrf.loss_graph()
	'''

	def __init__(self, minimum=1e-5, maximum=1e-2, step_size=None, n_epochs=None):
		super().__init__()
		self.minimum = minimum
		self.maximum = maximum
		self.i = 0
		self.num_iter = step_size * n_epochs
		self.history = {} # mandatory keras dictionary

	def calculate_learning_rate(self):
		x = self.i / self.num_iter
		return self.minimum + (self.maximum - self.minimum) * x

	# mandatory function for callbacks
	def on_train_begin(self, logs=None):
		if logs is None:
			logs = {}
		K.set_value(self.model.optimizer.lr, self.minimum)

	# mandatory function for callbacks
	def on_batch_end(self, epoch, logs=None):
		if logs is None:
			logs = {}

		self.i += 1

		self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
		self.history.setdefault('iterations', []).append(self.i)

		for key, value in logs.items():
			self.history.setdefault(key, []).append(value)

		K.set_value(self.model.optimizer.lr, self.calculate_learning_rate())

	# plot learning rate
	def lr_graph(self):
		plt.plot(self.history['iterations'], self.history['lr'])
		plt.yscale('log')
		plt.xlabel('i')
		plt.ylabel('lr')
		plt.show()

	# plot loss
	def loss_graph(self):
		plt.plot(self.history['lr'], self.history['loss'])
		plt.xscale('log')
		plt.xlabel('lr')
		plt.ylabel('loss')
		plt.show()

