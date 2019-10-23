import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


def load_set():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, v_set, t_set = pickle.load(f, encoding='bytes')
    f.close()

    return training_set, v_set, t_set


def get_x_and_label(set_type):
    x_set, label_set = set_type[0], set_type[1]

    x_set = np.array([np.reshape(item, 784) for item in x_set])

    return x_set, label_set


def set_weights():
    weight_set = np.random.rand(10, 784)
    return weight_set


def activation(threshold):
    output_array = np.zeros(10, dtype=int)
    i = 0
    for item in threshold:
        if item > 0:
            output_array[i] = 1
        else:
            output_array[i] = 0
        i += 1
    return output_array


def vectorized_label(label_value):
    label_array = np.zeros(10, dtype=int)
    label_array[label_value] = 1
    return label_array


def online_training(x_set, label_set, learning_rate, iterations):
    b = np.empty(10)
    b.fill(1.0)
    w = set_weights()
    for j in range(iterations):
        cnt = 0
        for k in x_set:
            z = np.dot(w, k) + b  # 10 perceptroni z.shape = (10, 1)
            output = activation(z)
            t = vectorized_label(label_set[cnt])
            copy_k = np.array([k])
            w += np.dot(np.transpose(np.array([(t - output)])), copy_k) * learning_rate  # inmultirea cu toate weight-urile deodata.
            b += (t - output) * learning_rate
            cnt += 1

    return w, b


def get_highest_output(x_set, weight_set, bias):
    a = np.dot(x_set, np.transpose(weight_set)) + bias
    return np.argmax(a, axis=1)


def get_accuracy(x_set, label_set, weight_set, a, bias):
    predicted = get_highest_output(x_set, weight_set, bias)
    correct = predicted == label_set
    accuracy = np.sum(correct) / float(a)
    print(f"{accuracy * float(100)}%")


if __name__ == "__main__":
    train_set, valid_set, test_set = load_set()
    x, label = get_x_and_label(train_set)

    weights, trained_bias = online_training(x, label, 0.1, 15)  # 0.00006 - 100 => 89% match
    copy_weight = weights

    x_test, label_test = get_x_and_label(test_set)

    print("Training set accuracy: ")
    get_accuracy(x, label, copy_weight, 50000, trained_bias)
    print("Testing set accuracy: ")
    get_accuracy(x_test, label_test, weights, 10000, trained_bias)
