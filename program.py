import neuralnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
	training_images = data['training_images']
	training_labels = data['training_labels']

layer_sizes = (784, 5, 10)

net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)




#import matplotlib.pyplot as plt

#plt.imshow(training_images[1].reshape(28,28), cmap = 'gray')
#plt.show()