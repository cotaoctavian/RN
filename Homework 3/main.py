import pickle
import gzip
import numpy as np


def load_set():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, v_set, t_set = pickle.load(f, encoding='bytes')
    f.close()

    return training_set, v_set, t_set


def get_x_values_and_label(set_type):
    x_set, label_set = set_type[0], set_type[1]

    x_set = np.array([np.reshape(item, 784) for item in x_set])

    return x_set, label_set


def initialize_weights(size):
    return [np.random.randn(j, i) / np.sqrt(i) for i, j in zip(size[0:], size[1:])]


def initialize_biases(size):
    return [np.random.randn(i) for i in size[1:]]


def vectorized_label(label_value):
    one_hot = np.zeros((10,))
    one_hot[label_value] = 1
    return one_hot


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def stochastic_gradient_descent(training_set, epochs, mini_batch_size, learning_rate, all_weights, all_biases):
    momentum = 0.5
    l2_reg = 0.1
    for i in range(epochs):
        print(f"Epoch number {i + 1}")
        index = np.arange(len(training_set[0]))
        np.random.shuffle(index)

        x_set = training_set[0][index]
        label_set = training_set[1][index]

        batches = list()
        for j in range(0, len(training_set[0]), mini_batch_size):
            new_mini_x = x_set[j: j + mini_batch_size]
            new_mini_label = label_set[j: j + mini_batch_size]
            batches.append((new_mini_x, new_mini_label))

        for mini_batch in batches:
            mini_batch_set = mini_batch[0]
            mini_label_set = mini_batch[1]
            counter = 0

            delta_3 = 0
            delta_2 = 0
            gradient_2_3 = 0
            gradient_1_2 = 0

            for x_value in mini_batch_set:
                data = backward_pass(x_value, mini_label_set[counter], learning_rate,
                                     all_weights, all_biases)

                delta_3 += momentum * learning_rate / len(mini_batch) * data['delta_3']
                delta_2 += momentum * learning_rate / len(mini_batch) * data['delta_2']
                gradient_2_3 += momentum * learning_rate / len(mini_batch) * data['gradient_2_3']
                gradient_1_2 += momentum * learning_rate / len(mini_batch) * data['gradient_1_2']

                counter += 1

            # Update weights + bias
            all_weights[1] = (1 - learning_rate * l2_reg / len(training_set[0])) * all_weights[1] - gradient_2_3
            all_weights[0] = (1 - learning_rate * l2_reg / len(training_set[0])) * all_weights[0] - gradient_1_2
            all_biases[1] -= delta_3
            all_biases[0] -= delta_2

    print("Training complete!")
    return all_biases, all_weights


def backward_pass(x_val, label_val, learning_rate, all_weights, all_biases):
    # Forward pass
    y_1 = x_val
    z_2 = np.dot(all_weights[0], x_val) + all_biases[0]
    y_2 = sigmoid(z_2)
    z_3 = np.dot(all_weights[1], y_2) + all_biases[1]
    y_3 = softmax(z_3)

    # Error cost for last layer
    delta_3 = y_3 - vectorized_label(label_val)

    # 2a
    delta_2 = np.dot(delta_3, all_weights[1]) * y_2 * (1 - y_2)

    # 2b
    gradient_2_3 = np.dot(np.array([delta_3]).T, np.array([y_2]))
    gradient_1_2 = np.dot(np.array([delta_2]).T, np.array([y_1]))

    # all_biases[1] = all_biases[1] - (delta_3 * learning_rate)
    # all_weights[1] = all_weights[1] - (learning_rate * gradient_2_3)
    # all_biases[0] = all_biases[0] - (delta_2 * learning_rate)
    # all_weights[0] = all_weights[0] - (learning_rate * gradient_1_2)

    return {"delta_3": delta_3, "delta_2": delta_2, "gradient_2_3": gradient_2_3, "gradient_1_2": gradient_1_2}
    # return all_biases, all_weights


def get_accuracy(x_val, label_val, a, all_weights, all_biases):
    correct = 0
    counter = 0
    for item in x_val:
        z_2 = np.dot(all_weights[0], item) + all_biases[0]
        y_2 = sigmoid(z_2)
        z_3 = np.dot(all_weights[1], y_2) + all_biases[1]
        y_3 = softmax(z_3)
        if label_val[counter] == np.argmax(y_3):
            correct += 1
        counter += 1

    accuracy = correct / float(a) * 100
    return accuracy


if __name__ == "__main__":
    train_set, validation_set, test_set = load_set()

    x, label = get_x_values_and_label(train_set)

    weights = initialize_weights([784, 100, 10])
    biases = initialize_biases([784, 100, 10])

    biases, weights = stochastic_gradient_descent(train_set, 5, 10, 0.1, weights, biases)

    open('result.txt', 'w').close()
    file = open("result.txt", "a")
    file.write("Accuracy on training set: ")
    file.write(str(get_accuracy(train_set[0], train_set[1], len(train_set[0]), weights, biases)))
    file.write(" %\n")
    file.write("-------------------------\n")
    file.write("Accuracy on validation set: ")
    file.write(str(get_accuracy(validation_set[0], validation_set[1], len(validation_set[0]), weights, biases)))
    file.write(" %\n")
    file.write("-------------------------\n")
    file.write("Accuracy on test set: ")
    file.write(str(get_accuracy(test_set[0], test_set[1], len(test_set[0]), weights, biases)))
    file.write(" %\n")
    file.close()
