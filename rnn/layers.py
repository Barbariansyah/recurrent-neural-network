import numpy as np
from typing import List, Tuple
from abc import ABC
from .common import relu, softmax, sigmoid
from math import ceil

__all__ = ['Layer', 'Dense']


class Layer(ABC):
    def call(self, inp: List[np.array]) -> List[np.array]:
        pass

    def calculate_output_shape(self, inp: List[tuple]):
        pass

class SimpleRNN(Layer):
    '''
    hidden size = number of neuron in hidden layer
    output size = number of neuron in output layer
    input shape <x, y>, x = number of timestamp, y = input dimension
    '''
    def __init__(self, hidden_size: int, output_size: int, input_shape: np.array, U: np.array = None, W: np.array = None, V: np.array = None):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_shape = input_shape
        self.U = U if U else np.random.uniform(-np.sqrt(1. / input_shape[1]), -np.sqrt(1. / input_shape[1]), (hidden_size, input_shape[1]))
        self.W = W if W else np.random.uniform(-np.sqrt(1. / hidden_size), -np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.V = V if V else np.random.uniform(-np.sqrt(1. / hidden_size), -np.sqrt(1. / hidden_size), (output_size, hidden_size))
        self.bxh = np.full(hidden_size, 0.1)
        self.bhy = np.full(output_size, 0.1)
        self.h = [np.zeros(hidden_size) for _ in range(input_shape[0] + 1)]
        self.out = [np.zeros(output_size) for _ in range(input_shape[0]  + 1)]

    
    def call(self, inp: List[np.array]) -> List[np.array]:
        for t, xt in enumerate(inp):
            Uxt = np.dot(self.U, xt)
            Wht_prev = np.dot(self.W, self.h[t]) + self.bxh
            ht = np.tanh(Uxt + Wht_prev)
            self.h[t+1] = ht

            Vht = np.dot(self.V, ht) + self.bhy
            yt = softmax(Vht)
            self.out[t+1] = yt


    def calculate_output_shape(self, inp: List[tuple]):
        return [self.input_shape[0], self.output_size]

    def __str__(self):
        for t in range(self.input_shape[0] + 1):
            print("t = {}".format(t))
            print(self.h[t])
            print(self.out[t])
    

class Dense(Layer):
    def __init__(self, unit_count: int, activation_function: str = 'relu'):
        self.unit_count = unit_count
        self.activation_function = activation_function
        self.last_layer = False

    def call(self, inp: List[np.array]) -> List[np.array]:
        result_dot_matrix = np.dot(inp[0], self.filters)
        result_dot_matrix = [np.add(result_dot_matrix, self.bias_weight)]
        result = self._activation(result_dot_matrix)

        return result

    def init_weight(self, input_size: List[tuple]):
        self.filters = np.random.random(
            (input_size[0][0], self.unit_count)) * 2 - 1
        self.bias_weight = np.random.random(
            self.unit_count)

    def _activation(self, conv_res: List[np.array]) -> List[np.array]:
        if self.activation_function == 'relu':
            reluv = np.vectorize(relu)
            result = [reluv(fm) for fm in conv_res]
        elif self.activation_function == 'softmax':
            result = softmax(conv_res)
        else:
            sigmoidv = np.vectorize(sigmoid)
            result = [sigmoidv(fm) for fm in conv_res]

        return result

    def calculate_output_shape(self, inp: List[tuple]):
        return [(self.unit_count,)]