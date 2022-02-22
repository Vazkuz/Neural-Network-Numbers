import neuralnetwork as nn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

with np.load('mnist.npz') as data:
	training_images = data['training_images']
	training_labels = data['training_labels']

layer_sizes = (784, 1000, 100, 10)

net = nn.NeuralNetwork(layer_sizes)
model = net.SetModel(layer_sizes)
keras.utils.plot_model(model, "C:/Users/Daniel/Desktop/Daniel M/_Proyectos/2022/NeuralNetworkNumbers/my_first_model.png", show_shapes=True)
model.compile(
	optimizer = tf.keras.optimizers.Adam(0.0001),
	loss = 'mean_squared_error'
)

# print(training_images[0])

print("Comenzando entrenamiento...")
historial = model.fit(training_images, training_labels, epochs=50, verbose = 1)
print("Modelo entrenado!")


import matplotlib.pyplot as plt
plt.xlabel('# Época')
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()




# import matplotlib.pyplot as plt

# plt.imshow(training_images[1].reshape(28,28), cmap = 'gray')
# plt.show()