import random
import numpy as np
import time

def cross_entropy_loss(y_hat, y, element_wise = False):
    delta = 1e-7
    
    if not element_wise:
        return -np.dot(y, np.log(y_hat + delta)) - np.dot((1-y), np.log(1-np.log(y_hat + delta)))
    loss = 0
    for i in range(m):
        loss += y[i] * np.log(y_hat[i] + delta) - (1-y[i]) * np.log(1-np.log(y_hat[i] + delta))
    
    return loss / m

def sigmoid(z, element_wise = False):

    if not element_wise:
        return 1 / (1 + np.exp(-z))
    for i in range(m):
        z[i] = 1 / (1 + np.exp(-z[i]))
    return z

def relu(x):
    return np.maximum(0., x)

def predict(w, b, x1_test, x2_test, y_test):
    result = sigmoid(w[0] * x1_test + w[1] * x2_test + b)
    result = np.where(result < 0.5, 0, 1)
    return result
    
def get_accuracy(y_pred, y_test):
    return np.mean(np.equal(y_pred, y_test)) * 100

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

def train_elements(w, b):

    z = np.zeros(m)
    for iter in range(k):

        # forward
        for i in range(m):
            z[i] = w[0] * x1_train[i] + w[1] * x2_train[i] + b
        z = np.array(z)
        a = sigmoid(z, element_wise=True)
        loss = cross_entropy_loss(a, y_train, element_wise=True)

        # backward
        dz = []
        dw1_sum, dw2_sum, dz_sum = 0, 0, 0
        for i in range(m):
            dz.append(a[i] - y_train[i])
            dw1_sum += x1_train[i] * dz[i]
            dw2_sum += x2_train[i] * dz[i]
            dz_sum += dz[i]
    
        dw1 = dw1_sum / m
        dw2 = dw2_sum / m
        db = dz_sum / m

        w[0] -= alpha * dw1
        w[1] -= alpha * dw2
        b -= alpha * db
        
        if iter % iter_print_value == 0:
            print('parameters : ', w, b)
            print('cost of train samples : ', loss)
            
            y_pred = predict(w, b, x1_train, x2_train, y_train)
            print('Accuracy for train sample : ', get_accuracy(y_pred, y_train), '%')

    return w, b

def train_vectorized(w, b):

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

        w[0] -= alpha * np.sum(dw1) / m
        w[1] -= alpha * np.sum(dw2) / m
        b -= alpha * db
 
        if iter % iter_print_value == 0:
            print('parameters : ', w, b)
            print('cost of train samples : ', loss)

            y_pred = predict(w, b, x1_train, x2_train, y_train)
            print('Accuracy for train sample : ', get_accuracy(y_pred, y_train), '%')

    return w, b

if __name__ == '__main__':

    # setting parameters
    x1_train, x2_train, y_train = [], [], []
    x1_test, x2_test, y_test = [], [], []

    iter_print_value = 10
    m = 1000
    n = 100
    k = 2000
    alpha = 0.01

    # prepare datasets
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = make_datasets()
    print('Check datasets : ', x1_train.shape, x2_train.shape, x1_test.shape, x2_test.shape, y_train.shape, y_test.shape)

    # w, b
    w = np.array([0.1, 0.1])
    b = 0.1

    time_consumption = time.process_time()
    w_experiment1, b_experiment1 = train_elements(w.copy(), b)
    time_experiment1 = time.process_time() - time_consumption

    time_consumption = time.process_time()
    w_experiment2, b_experiment2 = train_vectorized(w.copy(), b)
    time_experiment2 = time.process_time() - time_consumption
    
    
    y_pred_experiment1 = predict(w_experiment1, b_experiment1, x1_test, x2_test, y_test)
    print('-------------------------train_experiment1 end------------------------------')
    print('time - element.wise : ', time_experiment1, 's')
    print('parameter : ', w_experiment1, b_experiment1)
    print('cost of test samples : ', cross_entropy_loss(y_pred_experiment1, y_test))
    print('Accuracy for test sample : ', get_accuracy(y_pred_experiment1, y_test), '%')

    y_pred_experiment2 = predict(w_experiment2, b_experiment2, x1_test, x2_test, y_test)
    print('-------------------------train_experiment2 end------------------------------')
    print('time - element.wise : ', time_experiment2, 's')
    print('parameter : ', w_experiment2, b_experiment2)
    print('cost of test samples : ', cross_entropy_loss(y_pred_experiment2, y_test))
    print('Accuracy for test sample : ', get_accuracy(y_pred_experiment2, y_test), '%')