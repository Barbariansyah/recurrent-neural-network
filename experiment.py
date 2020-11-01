from rnn.layers import SimpleRNN, Dense
from rnn.rnn import MyRnn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
import math

'''
Forward Propagation Experiment
| Experiment | Sequence Length | Input Size | Hidden Size | Output Size | Initial Weight |
|      1     |         1       |     1      |      1      |      1      |        0       |
|      2     |         2       |     1      |      3      |      1      |        1       |
|      3     |         4       |     1      |      5      |      1      |      Random    |
'''

DATA_DIR = 'dataset/train_IBM.csv'
SCALER = MinMaxScaler(feature_range=(0, 1))


def load_dataset() -> List[np.array]:
    dataset = pd.read_csv(DATA_DIR, header=0)
    dataset = dataset["Close"].values
    train_dataset = dataset[:32]
    test_dataset = dataset[32:48]

    return train_dataset, test_dataset


def scale_data(dataset):
    dataset = np.reshape(dataset, (-1, 1))

    dataset = SCALER.fit_transform(dataset)
    temp_dataset = np.array([])
    for i in dataset:
        temp_dataset = np.append(temp_dataset, i[0])

    dataset = temp_dataset

    return dataset


def scale_data_inverse(dataset):
    dataset = np.reshape(dataset, (-1, 1))

    dataset = SCALER.inverse_transform(dataset)
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


def predict_sequence(model, initial_data, seq_length, step):
    final_result = []
    dataset = np.copy(initial_data)
    for i in range(seq_length):
        formatted_dataset = convert_to_matrix(dataset, step)
        test_x = formatted_dataset[0]
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        result = model.feed_forward(test_x)
        final_result.append(result[0][0])
        dataset = np.append(dataset, result[0][0])
        dataset = dataset[1:]

    return final_result


def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(np.mean(pow((y_pred - y_true), 2)))


if __name__ == "__main__":

    # Load train and test data
    train_dataset, test_dataset = load_dataset()

    # General preprocess
    train_dataset = scale_data(train_dataset)

    train_dataset = np.array(train_dataset)
    test_dataset = np.array(test_dataset)

    # Experiment 1
    SEQ_LENGTH_1 = 1
    INPUT_SIZE_1 = 1
    OUTPUT_SIZE_1 = 1
    HIDDEN_SIZE_1 = 1
    INIT_WEIGHT_1 = np.full((1, 1), 0)
    RETURN_SEQUENCES_1 = False

    # Preprocess data for experiment 1
    train_dataset_1 = np.append(train_dataset, np.repeat(
        train_dataset[-1, ], SEQ_LENGTH_1))
    test_dataset_1 = np.append(test_dataset, np.repeat(
        test_dataset[-1, ], SEQ_LENGTH_1))

    train_x_1, train_y_1 = convert_to_matrix(train_dataset_1, SEQ_LENGTH_1)
    test_x_1, test_y_1 = convert_to_matrix(test_dataset_1, SEQ_LENGTH_1)

    train_x_1 = np.reshape(
        train_x_1, (train_x_1.shape[0], 1, train_x_1.shape[1]))
    test_x_1 = np.reshape(test_x_1, (test_x_1.shape[0], 1, test_x_1.shape[1]))

    # Create model for experiment 1
    simple_rnn_1 = SimpleRNN(
        HIDDEN_SIZE_1, [SEQ_LENGTH_1, INPUT_SIZE_1], RETURN_SEQUENCES_1)
    simple_rnn_1.U = INIT_WEIGHT_1
    simple_rnn_1.W = INIT_WEIGHT_1
    rnn_1 = MyRnn()
    dense_1 = Dense(OUTPUT_SIZE_1)
    rnn_1.add(simple_rnn_1)
    rnn_1.add(dense_1)

    for data in train_x_1:
        ff_result_1, _ = rnn_1.feed_forward(data[0])
    ff_result_1 = scale_data_inverse(ff_result_1[0])

    print('Experiment 1')
    print(rnn_1)
    print('Output Model : {} \n'.format(ff_result_1))


    predict_result = predict_sequence(rnn_1, train_dataset[-51:], test_dataset.shape[0], 1)
    rmse_result = root_mean_squared_error(test_dataset, predict_result)
    print('RMSE Experiment 1 : {} \n'.format(rmse_result))
    print('=====================')

    # Experiment 2
    SEQ_LENGTH_2 = 2
    INPUT_SIZE_2 = 1
    OUTPUT_SIZE_2 = 1
    HIDDEN_SIZE_2 = 3
    INIT_U_WEIGHT_2 = np.full((HIDDEN_SIZE_2, INPUT_SIZE_2), 1)
    INIT_W_WEIGHT_2 = np.full((HIDDEN_SIZE_2, HIDDEN_SIZE_2), 1)
    RETURN_SEQUENCES_2 = False

    # Preprocess data for experiment 2
    train_dataset_2 = np.append(train_dataset, np.repeat(
        train_dataset[-1, ], SEQ_LENGTH_2))
    test_dataset_2 = np.append(test_dataset, np.repeat(
        test_dataset[-1, ], SEQ_LENGTH_2))

    train_x_2, train_y_2 = convert_to_matrix(train_dataset_2, SEQ_LENGTH_2)
    test_x_2, test_y_2 = convert_to_matrix(test_dataset_2, SEQ_LENGTH_2)

    train_x_2 = np.reshape(
        train_x_2, (train_x_2.shape[0], 1, train_x_2.shape[1]))
    test_x_2 = np.reshape(test_x_2, (test_x_2.shape[0], 1, test_x_2.shape[1]))

    # Create model for experiment 2
    simple_rnn_2 = SimpleRNN(
        HIDDEN_SIZE_2, [SEQ_LENGTH_2, INPUT_SIZE_2], RETURN_SEQUENCES_2, U=INIT_U_WEIGHT_2, W=INIT_W_WEIGHT_2)
    rnn_2 = MyRnn()
    dense_2 = Dense(OUTPUT_SIZE_2)
    rnn_2.add(simple_rnn_2)
    rnn_2.add(dense_2)

    for data in train_x_2:
        ff_result_2 = rnn_2.feed_forward(data[0])
    ff_result_2 = scale_data_inverse(ff_result_2[0])

    print('Experiment 2')
    print(rnn_2)
    print('Output Model : {} \n'.format(ff_result_2))

    predict_result = predict_sequence(rnn_2, train_dataset[-51:], test_dataset.shape[0], 1)
    rmse_result = root_mean_squared_error(test_dataset, predict_result)
    print('RMSE Experiment 2 : {} \n'.format(rmse_result))
    print('=====================')

    # Experiment 3
    SEQ_LENGTH_3 = 4
    INPUT_SIZE_3 = 1
    OUTPUT_SIZE_3 = 1
    HIDDEN_SIZE_3 = 5
    RETURN_SEQUENCES_3 = False

    # Preprocess data for experiment 2
    train_dataset_3 = np.append(train_dataset, np.repeat(
        train_dataset[-1, ], SEQ_LENGTH_3))
    test_dataset_3 = np.append(test_dataset, np.repeat(
        test_dataset[-1, ], SEQ_LENGTH_3))

    train_x_3, train_y_3 = convert_to_matrix(train_dataset_3, SEQ_LENGTH_3)
    test_x_3, test_y_3 = convert_to_matrix(test_dataset_3, SEQ_LENGTH_3)

    train_x_3 = np.reshape(
        train_x_3, (train_x_3.shape[0], 1, train_x_3.shape[1]))
    test_x_3 = np.reshape(test_x_3, (test_x_3.shape[0], 1, test_x_3.shape[1]))

    # Create model for experiment 3
    rnn_3 = MyRnn()
    dense_3 = Dense(OUTPUT_SIZE_3)
    simple_rnn_3 = SimpleRNN(HIDDEN_SIZE_3, [SEQ_LENGTH_3, INPUT_SIZE_3])
    rnn_3.add(simple_rnn_3)
    rnn_3.add(dense_3)

    for data in train_x_3:
        ff_result_3 = rnn_3.feed_forward(data[0])
    ff_result_3 = scale_data_inverse(ff_result_3[0])

    print('Experiment 3')
    print(rnn_3)
    print('Output Model : {} \n'.format(ff_result_3))

    predict_result = predict_sequence(rnn_3, train_dataset[-51:], test_dataset.shape[0], 1)
    rmse_result = root_mean_squared_error(test_dataset, predict_result)
    print('RMSE Experiment 3 : {} \n'.format(rmse_result))
    print('=====================')
