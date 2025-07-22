import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt(os.path.join('/Users/liupuchen/VsCode/PythonProject/LinearRegression-study/ex1data2.txt'), delimiter=',')
X=data[:,:2]
y=data[:,2]
m=len(y)

def feature_normalize(X):
    mu=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm=(X-mu)/std
    return X_norm,mu,std

def compute_cost(X,y,theta):
    J=1/(2*m)*np.sum(np.square(X@theta-y))
    return J

def gradient_descent(X,y,theta,alpha,iter):
    J_hist=np.zeros(iter)
    m=len(y)
    theta=theta.copy()
    for i in range(iter):
        theta=theta-alpha*(1/m)*(X.T@X@theta-X.T@y)
        J_hist[i]=compute_cost(X,y,theta)
    return theta,J_hist

X_norm,mu_X,std_X=feature_normalize(X)
X_norm=np.concatenate((np.ones((m,1)),X_norm),axis=1)
y_norm,mu_y,std_y=feature_normalize(y)
theta=np.zeros(X_norm.shape[1])
alpha=0.1
iter=500
theta,J_hist=gradient_descent(X_norm,y_norm,theta,alpha,iter)
theta[1:]=(theta[1:]*std_y)/std_X
theta[0]=theta[0]*std_y+mu_y-np.sum(theta[1:]*mu_X)
print(theta)
price=[1.0,1650.0,3.0]@theta
print(price)
plt.plot(range(iter),J_hist,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()