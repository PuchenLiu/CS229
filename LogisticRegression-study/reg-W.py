import os
import numpy as np

data=np.loadtxt(os.path.join('/Users/liupuchen/VsCode/PythonProject/LogisticRegression-study/ex2data2.txt'),delimiter=',')
X=data[:,:2]
y=data[:,2]

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunctionReg(theta,X,y,lamda):
    m=len(y)
    z=X@theta
    h=sigmoid(z)
    g=np.zeros(len(theta))
    beta=theta
    beta[0]=0
    J=(-1/(m))*(y.T@np.log(h)+(1-y).T@(np.log(1-h)))+(lamda/(2*m))*theta.T@beta
    g=(1/m)*X.T@(h-y)+(lamda/m)*beta
    return J,g

def mapFeature(X1,X2,degree):
    out=[]
    for i in range(degree+1):
        for j in range(degree+1-i):
            out.append((X1**(i))*(X2**j))
    out=np.array(out).T
    return out

def predict(theta,X):
    p=sigmoid(X@theta)
    return p>0.5

def gradientDescent(X,y,theta,alpha,iter,lamda):
    m=len(y)
    J_history=[]
    J,g=costFunctionReg(theta,X,y,lamda)
    for i in range(iter):
        theta=theta-alpha*g
        J,g=costFunctionReg(theta,X,y,lamda)
        J_history.append(J)
    return theta,J_history

X=mapFeature(X[:,0],X[:,1],6)
m,n=X.shape
initial_theta=np.zeros(n)
lamda=1
alpha=0.5
cost,grad=costFunctionReg(initial_theta,X,y,lamda)
print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')