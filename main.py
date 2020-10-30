from rnn.layers import SimpleRNN

if __name__ == "__main__":
    rnn = SimpleRNN(3, 4, [10,4])
    print(rnn.U)
    print(rnn.V)
    print(rnn.W)
    print(rnn.bxh)
    print(rnn.bhy)