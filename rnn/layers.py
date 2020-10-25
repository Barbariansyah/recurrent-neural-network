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
    def __init__(self, units: int, output_shape: np.array):
        pass

    def call(self, inp: List[np.array]) -> List[np.array]:
        pass

    def calculate_output_shape(self, inp: List[tuple]):
        pass

    def __str__(self):
        pass
    

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