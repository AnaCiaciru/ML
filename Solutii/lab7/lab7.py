import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)


def plot_decision_boundary(X, y, W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)

    # sterge continutul ferestrei
    plt.clf()

    # ploteaza multimea de antrenare
    color = 'r'

    if (current_y == -1):
        color = 'b'

    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')

    # ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color + 's')

    # afisarea dreptei de decizie
    plt.plot([x1, x2], [y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)


def train_perc(train_data, train_labels, epochs, lr):
    # step 1
    w = np.zeros(2)
    b = 0
    no_samps = x.shape[0] # step 2
    acc = 0.0  # step 3
    # step 4
    for epoch in range(epochs):
        # 4.1
        train_data, train_labels = shuffle(train_data, train_labels)
        # 4.2
        for idx in range(no_samps):
            y_hat = np.dot(train_data[idx][:], w) + b  # predict
            loss = (y_hat - train_labels[idx]) ** 2  # compute loss
            w -= lr * (y_hat - train_labels[idx]) * train_data[idx][:] # update weights
            b -= lr * (y_hat - train_labels[idx]) # update bias
            acc = np.mean(np.sign(np.dot(train_data, w) + b) == train_labels)
            print(loss, acc)
            plot_decision_boundary(train_data, train_labels, w, b, train_data[idx][:], train_labels[idx])
    return(w, b)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, 1])
epochs = 70
lr = 0.1

w, b = train_perc(x, y, epochs, lr)

print(w)
print(b)