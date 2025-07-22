import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 防止溢出
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmaxCostFun(X, y, theta, lambda_, num_labels):
    m = X.shape[0]
    theta = theta.reshape((X.shape[1], num_labels))
    logits = X @ theta
    probs = softmax(logits)

    y_matrix = np.eye(num_labels)[y]  # one-hot 编码

    cost = -np.sum(y_matrix * np.log(probs)) / m
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:, :]))
    J = cost + reg
    grad = (X.T @ (probs - y_matrix)) / m
    grad[1:, :] += (lambda_ / m) * theta[1:, :]  # 正则化不作用于偏置项

    return J, grad

def GraDesc(X, y, theta, iter, alpha, lambda_, num_labels):
    J_his = []
    alpha=0.001
    beta1=0.9
    beta2=0.999
    sigma=10**(-8)
    m=0
    v=0
    for i in range(iter):
        J, grad = softmaxCostFun(X, y, theta, lambda_, num_labels)
        m=(beta1*m+(1-beta1)*grad)
        v=(beta2*v+(1-beta2)*(grad**2))
        theta=theta-alpha*(((1/(1-beta1**(i+1)))*m)/(np.sqrt(((1/(1-beta2**(i+1)))*v))+sigma))
        J_his.append(J)
    return theta, J_his

def predict(X, theta):
    logits = X @ theta
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

# ========== 主程序 ==========

input_size = 400
num_labels = 10
data = loadmat('/Users/liupuchen/VsCode/PythonProject/OVR-LogisticRegression-study/ex3data1.mat')
X, y = data['X'], data['y'].flatten()
X = np.hstack((np.ones((X.shape[0], 1)), X))
y[y == 10] = 0  # 将标签 10 转换为 0，匹配索引

theta = np.zeros((X.shape[1], num_labels))  # 初始化参数矩阵
lambda_ = 0.1
alpha = 0.1
iter = 6000

theta, J_history = GraDesc(X, y, theta, iter, alpha, lambda_, num_labels)

pred = predict(X, theta)
accuracy = np.mean(pred == y) * 100
print('Training Set Accuracy: {:.2f}%'.format(accuracy))