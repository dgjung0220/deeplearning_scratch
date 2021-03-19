import random
import numpy as np

def cross_entropy_loss(y_hat, y):
    return -np.dot(y, np.log(y_hat)) - np.dot((1-y), np.log(1-np.log(y_hat)))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(x):
    return np.maximum(0., x)

def predict(w, b, x1_test, x2_test, y_test):
    result = sigmoid(w[0] * x1_test + w[1] * x2_test + b)
    return result

def get_accuracy(y_pred, y_test):
    return np.mean(np.equal(y_pred, y_test))

def make_datasets():
    
    # datasets for train
    for i in range(m):
        x1_train.append(random.uniform(-10, 10))
        x2_train.append(random.uniform(-10, 10))

        if x1_train[-1] + x2_train[-1] > 0:
            y_train.append(1)
        else:
            y_train.append(0)

    # dataset for test
    for i in range(n):
        x1_test.append(random.uniform(-10, 10))
        x2_test.append(random.uniform(-10, 10))

        if x1_test[-1] + x2_test[-1] > 0:
            y_test.append(1)
        else:
            y_test.append(0)

    return np.array(x1_train), np.array(x2_train), np.array(x1_test), np.array(x2_test), np.array(y_train), np.array(y_test)

if __name__ == '__main__':

    # test()
    # setting parameters
    x1_train, x2_train, y_train = [], [], []
    x1_test, x2_test, y_test = [], [], []

    iter_print_value = 10
    m = 1000
    n = 100
    k = 2000
    alpha = 0.0001

    # prepare datasets
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = make_datasets()
    print(x1_train.shape, x2_train.shape, x1_test.shape, x2_test.shape, y_train.shape, y_test.shape)

    # w, b
    w = [0.1, 0.1]
    b = 0.2

    # train
    for iter in range(k):
        
        # forward
        z = w[0] * x1_train + w[1] * x2_train + b
        a = sigmoid(z)
        loss = cross_entropy_loss(a, y_train)
            
        # backward
        dz = a - y_train
        dw1 = np.dot(x1_train, dz)
        dw2 = np.dot(x2_train, dz)
        db = np.sum(dz) / m

        w[0] -= alpha * dw1
        w[1] -= alpha * dw2
        b -= alpha * db
 
        if iter % iter_print_value == 0:
            print('parameters : ', w, b)
            print('cost of train samples : ', loss)
            # print(predict(w, b, x1_train, x2_train, y_train))
            print('Accuracy for train sample : ', get_accuracy(predict(w, b, x1_train, x2_train, y_train), y_train), '%')
    

    y_pred = predict(w, b, x1_test, x2_test, y_test)
    print('-------------------------train end------------------------------')
    print('cost of test samples : ', cross_entropy_loss(y_pred, y_test))
    print('Accuracy for test sample : ', get_accuracy(y_pred, y_test), '%') 