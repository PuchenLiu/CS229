import os
import numpy as np

data=np.loadtxt(os.path.join('/Users/liupuchen/VsCode/PythonProject/LogisticRegression-study/ex2data2.txt'),delimiter=',')
X=data[:,:2]
y=data[:,2]

def sigmoid(X,theta):
    return 1/(1+np.exp(-X@theta))

def costFunctionReg(X,y,theta,lamda,m):
    h=sigmoid(X,theta)
    J=(-1/m)*(y.T@np.log(h)+(1-y).T@(np.log(1-h)))+lamda/(2*m)*theta.T@theta
    grad=(-1/m)*X.T@(h-y)+lamda/m*theta
    return J,grad

def mapFeature(X1,X2,degree):
    out=[np.ones(X1.shape[0])]
    for i in range(1,degree+1):
        for j in range(i+1):
            out.append((X1**(i-j))*(X2**j))
    return np.array(out).T

def gradDesc(X,y,theta,alpha,iter,lamda,m):
    J_history=[]
    h=sigmoid(X,theta)
    J,grad=costFunctionReg(X,y,theta,lamda,m)
    for i in range(iter):
        theta=theta-alpha*grad
        J,grad=costFunctionReg(X,y,theta,lamda,m)
        J_history.append(J)
    return theta,J_history

X=mapFeature(X[:,0],X[:,1],6)
m,n=X.shape
initial_theta=np.zeros(n)
lamda=1
alpha=0.5
cost,grad=costFunctionReg(X,y,initial_theta,lamda,m)
print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(X, y,test_theta,10,m)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')