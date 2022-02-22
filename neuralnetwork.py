import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# inputs = keras.Input(shape=(784,))
# dense = layers.Dense(28, activation="relu")
# x = dense(inputs)
# x = layers.Dense(3, activation="relu")(x)
# x = layers.Dense(1, activation="relu")(x)
# x = layers.Dense(2, activation="relu")(x)
# outputs = layers.Dense(10)(x)

# model = keras.Sequential

# keras.utils.plot_model(model, "C:/Users/Daniel/Desktop/Daniel M/_Proyectos/2022/NeuralNetworkNumbers/my_first_model.png", show_shapes=True)



class NeuralNetwork:
    
    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    def predict(self, a):
        for w,b in zip(self.weights, self.biases):
            a = self.sigmoid(np.matmul(w, a) + b)
        return a

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b) for a,b in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct, len(images), (num_correct/len(images))*100))

    def SetModel(self, layer_sizes):
        hidden_layers = []
        hidden_layers.append(tf.keras.layers.Dense(units = layer_sizes[1], input_shape = [layer_sizes[0]]))
        for layer_idx in range(2, len(layer_sizes)):
            hidden_layers.append(tf.keras.layers.Dense(units = layer_sizes[layer_idx]))
        return tf.keras.Sequential([layer for layer in hidden_layers])


    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
        


    
