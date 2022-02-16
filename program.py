import neuralnetwork as nn
import numpy as np
#import matplotlib.pyplot as plt

with np.load('mnist.npz') as data:
	training_images = data['training_images']
	training_labels = data['training_labels']

plt.imshow(training_images[1].reshape(28,28), cmap = 'gray')
plt.show()

print(training_labels[1])
print(training_labels.shape)

np.datetime_as_string 

layer_sizes = (3, 5, 10)
x = np.ones((layer_sizes[0], 1))

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)

print(prediction)