import numpy as np
from typing import List, Tuple
from abc import ABC
from .common import relu, softmax, sigmoid
from math import ceil

__all__ = ['Layer', 'SimpleRnn', 'Dense']


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

    def __init__(self, hidden_size: int, input_shape: np.array, return_sequence: bool = False, U: np.array = None, W: np.array = None):
        self.hidden_size = hidden_size
        self.input_shape = input_shape
        self.return_sequence = return_sequence
        self.U = U if np.array(U).any() else np.random.uniform(-np.sqrt(
            1. / input_shape[1]), -np.sqrt(1. / input_shape[1]), (hidden_size, input_shape[1]))
        self.W = W if np.array(W).any() else np.random.uniform(-np.sqrt(
            1. / hidden_size), -np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.bxh = np.full(hidden_size, 0.1)

    def call(self, inp: List[np.array]) -> List[np.array]:
        self.h = [np.zeros(self.hidden_size)]
        for t in range(self.input_shape[0]):
            Uxt = np.dot(self.U, inp[t])
            Wht_prev = np.dot(self.W, self.h[-1]) + self.bxh
            ht = np.tanh(Uxt + Wht_prev)
            self.h.append(ht)

        if self.return_sequence:
            return self.h
        else:
            return self.h[-1]

    def calculate_output_shape(self, inp: List[tuple]):
        return [(self.input_shape[0], self.hidden_size)] if self.return_sequence else [(self.hidden_size,)]

    def __str__(self):
        res = ''
        for t, ht in enumerate(self.h):
            res += "t = {} ht = {} \n".format(t, ht)
        return res


class Dense(Layer):
    def __init__(self, unit_count: int, activation_function: str = 'relu'):
        self.unit_count = unit_count
        self.activation_function = activation_function
        self.last_layer = False
        self.output = None

    def call(self, inp: List[np.array]) -> List[np.array]:
        result_dot_matrix = np.dot(inp[0], self.filters)
        result_dot_matrix = [np.add(result_dot_matrix, self.bias_weight)]
        result = self._activation(result_dot_matrix)

        self.output = result
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
        elif self.activation_function == 'sigmoid':
            sigmoidv = np.vectorize(sigmoid)
            result = [sigmoidv(fm) for fm in conv_res]
        else:
            linearv = np.vectorize(linear)
            result = [linearv(fm) for fm in conv_res]

        return result

    def calculate_output_shape(self, inp: List[tuple]):
        return [(self.unit_count,)]

    def __str__(self):
        res = "output dense = {} \n".format(self.output)
        return res
