import numpy as np
from typing import List


def relu(x: float) -> float:
    return x if x > 0 else 0


def softmax(inp: List[np.array]) -> list:
    x = inp[0]
    e_x = np.exp(x - np.max(x))
    res = e_x / e_x.sum()
    return [res]


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-1*x))

def linear(x: float) -> float:
    return x