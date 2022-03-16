import neuralnetwork as nn
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from PIL import Image

with np.load('mnist.npz') as data:
	training_images = data['training_images']
	training_labels = data['training_labels']

layer_sizes = (784, 500, 100, 10)

to_predict = 6

net = nn.NeuralNetwork(layer_sizes)
model = net.SetModel(layer_sizes)

model.compile(
	optimizer = tf.keras.optimizers.Adam(1e-5),
	loss = 'categorical_crossentropy',
	metrics = ['categorical_accuracy']
)

print("Comenzando entrenamiento...")
historial = model.fit(training_images,training_labels, batch_size = 250, epochs = 150)
print("Modelo entrenado!")


# import matplotlib.pyplot as plt
# plt.xlabel('# Época')
# plt.ylabel("Magnitud de pérdida")
# plt.plot(historial.history["loss"])
# plt.show()

im = Image.open('C:/Users/Daniel/Desktop/Daniel M/_Proyectos/2022/NeuralNetworkNumbers/prueba_{0}.png'.format(to_predict), 'r')
im = im.convert("L")
pix_val = np.array(list(im.getdata()))/255.

imagen = pix_val
imagen = imagen.reshape(1,784)

prediccion = model.predict(imagen)
print("Predicción: " + 	str(np.argmax(prediccion[0])))


import matplotlib.pyplot as plt

plt.imshow(pix_val.reshape(28,28), cmap = 'gray')
plt.show()