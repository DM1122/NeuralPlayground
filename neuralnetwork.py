import numpy as np


class NeuralNetwork:

    def __init__(self, shape):
        '''
        Creates a neural net instance.

        size (tuple): network size and shape. 

        '''
        self.shape = shape
        self.nodes = sum(shape)

        self.weight_shapes = [(a,b) for a,b in zip(shape[1:],shape[:-1])]
        self.weights = [np.random.standard_normal(s) for s in self.weight_shapes]
        self.biases = [np.zeros((s,1)) for s in shape[1:]]


    def __str__(self):
        string = '{} ({}) \nShape: {} \nNodes: {} \nWeights: {}'.format(type(self), id(self), self.shape, self.nodes, self.weight_shapes)

        return string


    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))


    def predict(self, a):
        for w,b in zip(self.weights, self.biases):
            z = np.matmul(w,a) + b
            a = self.activation(z)
        return a






if __name__ == '__main__':
    shape = (2,3,1)
    model = NeuralNetwork(shape)

    x = np.ones((2,1))
    prediction = model.predict(x)

    print(prediction)