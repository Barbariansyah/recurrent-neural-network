from rnn.layers import SimpleRNN, Dense
from rnn.rnn import MyRnn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List

'''
Forward Propagation Experiment
| Experiment | Sequence Length | Input Size | Hidden Size | Output Size | Initial Weight |
|      1     |         8       |     1      |      1      |      1      |        0       |
|      2     |         16      |     1      |      3      |      1      |        1       |
|      3     |         32      |     1      |      5      |      1      |      Random    |
'''

DATA_DIR = 'dataset/train_IBM.csv'


def load_dataset() -> List[np.array]:
    dataset = pd.read_csv(DATA_DIR, header=0)
    dataset = dataset["Close"].values
    train_dataset = dataset[:32]
    test_dataset = dataset[32:48]

    return train_dataset, test_dataset


def scale_data(dataset):
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    temp_dataset = np.array([])
    for i in dataset:
        temp_dataset = np.append(temp_dataset, i[0])

    dataset = temp_dataset

    return dataset


def convert_to_matrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i+step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)


if __name__ == "__main__":

    # Load train and test data
    train_dataset, test_dataset = load_dataset()
    # Preprocess train data
    train_dataset = scale_data(train_dataset)

    # train_dataset = np.array(train_dataset)
    # test_dataset = np.array(test_dataset)

    # test_dataset = np.append(test_dataset, np.repeat(test_dataset[-1, ], 1))
    # train_dataset = np.append(train_dataset, np.repeat(train_dataset[-1, ], 1))

    # train_x, train_y = convert_to_matrix(train_dataset, 1)
    # test_x, test_y = convert_to_matrix(test_dataset, 1)

    # train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    # test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    # Experiment 1
    init_weight_1 = np.full((1, 1), 0)
    simple_rnn_1 = SimpleRNN(
        1, 1, [32, 1])
    simple_rnn_1.U = init_weight_1
    simple_rnn_1.V = init_weight_1
    simple_rnn_1.W = init_weight_1
    rnn_1 = MyRnn()
    rnn_1.add(simple_rnn_1)
    rnn_1.add(Dense(1))

    rnn_1.feed_forward(train_dataset)
    print('Experiment 1')
    print(simple_rnn_1)
    print('=====================')

    # Experiment 2
    init_weight_2 = np.full((1, 1), 1)
    simple_rnn_2 = SimpleRNN(
        1, 1, [32, 1], init_weight_2, init_weight_2, init_weight_2)
    rnn_2 = MyRnn()
    rnn_2.add(simple_rnn_2)
    rnn_2.add(Dense(1))

    rnn_2.feed_forward(train_dataset)
    print('Experiment 2')
    print(simple_rnn_2)
    print('=====================')

    # Experiment 3
    rnn_3 = MyRnn()
    simple_rnn_3 = SimpleRNN(1, 1, [32, 1])
    rnn_3.add(simple_rnn_3)
    rnn_3.add(Dense(1))

    rnn_3.feed_forward(train_dataset)
    print('Experiment 3')
    print(simple_rnn_3)
    print('=====================')
