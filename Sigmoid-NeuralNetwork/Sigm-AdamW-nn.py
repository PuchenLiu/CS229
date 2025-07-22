import os
import numpy as np
from scipy.io import loadmat

def sigmoid(z):
    return 1/(1+np.exp(-z))

def nnCostFun(theta1,theta2,num_label,X,y,lambda_):
    m=X.shape[0]
    y=np.eye(num_label)[y]

    z1=X@theta1.T
    a1=sigmoid(z1)
    a1=np.concatenate([np.ones((m,1)),a1],axis=1)
    z2=a1@theta2.T
    a2=sigmoid(z2)
    J=np.sum((-1/m)*(y*np.log(a2)+(1-y)*np.log(1-a2)))+(lambda_/(2*m))*(np.sum(theta1[:,1:]*theta1[:,1:])+np.sum(theta2[:,1:]*theta2[:,1:]))
    delta2=a2-y
    delta1=delta2@theta2[:,1:]*a1[:,1:]*(1-a1[:,1:])
    theta2_grad=(1/m)*delta2.T@a1
    theta1_grad=(1/m)*delta1.T@X
    
    return J,theta1_grad,theta2_grad

def GradDesc(X,y,lambda_,num_label,theta1,theta2,iter):
    J_his=[]
    alpha=0.001
    beta1=0.9
    beta2=0.999
    sigma=10**(-8)
    m=X.shape[0]
    a,b=theta1.shape
    c,d=theta2.shape
    m1=np.zeros((a,b))
    v1=np.zeros((a,b))
    m2=np.zeros((c,d))
    v2=np.zeros((c,d))
    for i in range(iter):
        J,theta1_grad,theta2_grad=nnCostFun(theta1,theta2,num_label,X,y,lambda_)
        m2=beta1*m2+(1-beta1)*theta2_grad
        v2=beta2*v2+(1-beta2)*(theta2_grad**2)
        m2_=m2/(1-beta1**(i+1))
        v2_=v2/(1-beta2**(i+1))
        theta2=theta2-alpha*(m2_/(np.sqrt(v2_)+sigma))
        theta2[:,1:]-=alpha*(lambda_/m)*theta2[:,1:]
        
        m1=beta1*m1+(1-beta1)*theta1_grad
        v1=beta2*v1+(1-beta2)*(theta1_grad**2)
        m1_=m1/(1-beta1**(i+1))
        v1_=v1/(1-beta2**(i+1))
        theta1=theta1-alpha*(m1_/(np.sqrt(v1_)+sigma))
        theta1[:,1:]-=alpha*(lambda_/m)*theta1[:,1:]
        
        J_his.append(J)

    return J_his,theta1,theta2

def RandInitializeW(L_in,L_out,epsilon):
    #w=np.zeros((L_out,L_in+1))
    w=np.random.rand(L_out,L_in+1)*2*epsilon-epsilon

    return w

def predict(X,theta1,theta2):
    a1=sigmoid(X@theta1.T)
    a1=np.hstack((np.ones((a1.shape[0], 1)), a1))
    a2=sigmoid(a1@theta2.T)
    p=np.argmax(a2,axis=1)
    return p

input_size = 400
hidden_size=25
num_label = 10
lambda_=1
iter=5000
epsilon=0.12
data = loadmat('/Users/liupuchen/VsCode/PythonProject/OVR-LogisticRegression-study/ex3data1.mat')
X, y = data['X'], data['y'].flatten()
X = np.hstack((np.ones((X.shape[0], 1)), X))
y[y == 10] = 0 

theta1=RandInitializeW(input_size,hidden_size,epsilon)
theta2=RandInitializeW(hidden_size,num_label,epsilon)
J_his,theta1,theta2=GradDesc(X,y,lambda_,num_label,theta1,theta2,iter)
p=predict(X,theta1,theta2)
acc=np.mean(p==y)*100
print("Training accuracy: %.2f%%" % acc)