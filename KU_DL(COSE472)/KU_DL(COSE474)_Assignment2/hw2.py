'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# cross entropy loss
########################################


'''
n: data size
num_class: number of classes
feat_dim: feature dimension : will be set to 2
Wb : num_class * feat_dim + num_class
'''


# returns the cross-entropy loss (should be nonnegative), averaged over the dataset.
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    '''
    Wb = np.reshape(Wb, (num_class, feat_dim + 1))
    bias_term = np.ones((n, 1))
    x = np.concatenate((x, bias_term), axis=1)
    s = x@Wb.T

    주석 내에 있는 코드는 잘못 작성된 코드인데, 후에 공부하려고 남겨두었습니다. 무시하고 채점해주시면 감사하겠습니다.
    '''

    Wb = np.reshape(Wb, (-1, 1))  # flatten
    b = Wb[-num_class:].squeeze()  # bias 분리
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    s = x@W.T + b

    sum_list = np.ndarray((n))

    for i in range(n):
        prevent_overflow_term = np.max(s[i])
        sum_list[i] = np.sum(np.exp(s[i] - prevent_overflow_term))
        s[i] = np.exp(s[i] - prevent_overflow_term) / sum_list[i]

    loss = 0
    for i in range(n):
        loss += -np.log(s[i, y[i]])

    return loss / n


# now lets test the model for linear models, that is, SVM and softmax


def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]

    Wb = np.reshape(Wb, (-1, 1))  # flatten
    b = Wb[-num_class:].squeeze()  # bias 분리
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x@W.T + b
    # score has shape (n_test, num_class)
    # (feat_dim * n_test) @ (num_class * feat_dim).T

    # get argmax over class dim
    # 엔트로피 함수에 의해서 각 class에 대한 확률이 나옴.
    res = np.argmax(s, axis=1)

    # get accuracy
    accuracy = (res == y).astype('uint8').sum()/n_test

    return accuracy


# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ', num_class, ' sigma for data scatter:', sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(
    number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(
    number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# start training softmax classifier
print('training softmax classifier...')
w0 = np.random.normal(0, 1, (2 * num_class + num_class))
# feature : fix to 2
# then, W0 = 3 num_class
result = minimize(cross_entropy_softmax_loss, w0, args=(
    x_train, y_train, num_class, n_train, feat_dim))

print('testing softmax classifier...')

Wb = result.x
print('accuracy of softmax loss: ', linear_classifier_test(
    Wb, x_test, y_test, num_class)*100, '%')
