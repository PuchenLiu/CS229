import os
import numpy as np  
from matplotlib import pyplot as plt
from scipy.io import loadmat

def sigmoid(z):
    return 1/(1+np.e**(-z))

def costFunction(theta, X, y,lambda_):
    m= len(y)
    h = sigmoid(X @ theta)
    beta = theta.copy()
    beta[0] = 0
    j=(-1/m)*(y.T@np.log(h)+(1-y).T@np.log(1-h))+(lambda_/(2*m)*beta.T@beta)
    g=(1/m)*X.T@(h-y)+ (lambda_/m)*beta
    return j,g

def GradientDescent(theta,X,y,alpha,iter,lambda_):
    J_history=[]
    J,g=costFunction(theta,X,y,lambda_)
    for i in range(iter):
        theta=theta-alpha*g
        J,g=costFunction(theta,X,y,lambda_)
        J_history.append(J)
    return theta,J_history

def OVR(X,y,num_labels,lambda_,iter,alpha):
    m,n=X.shape
    all_theta=np.zeros((n, num_labels))
    all_J_history = []
    for i in range(num_labels):
        theta= np.zeros(n)
        k = (y == i).astype(int)
        theta,J_history=GradientDescent(theta,X,k,alpha,iter,lambda_)
        all_theta[:,i]=theta
        all_J_history.append(J_history)
    return all_theta, all_J_history
    
def predict(all_theta,X):
    m=X.shape[0]
    p=sigmoid(X@all_theta)
    a= np.zeros((m,1))
    for i in range(m):
        k=0
        for j in range(num_labels):
            if(p[i,j]>=p[i,k]):
                k=j
        a[i]=k    
    return a




imput_layer_size=400
num_labels=10
data= loadmat(os.path.join('/Users/liupuchen/VsCode/PythonProject/OVR-LogisticRegression-study/ex3data1.mat'))
X,y=data['X'],data['y'].flatten()
X=np.hstack((np.ones((X.shape[0],1)),X)) 
y[y==10]=0
lambda_=0.1
all_theta, all_J_history = OVR(X, y, num_labels, lambda_, 1000, 0.1)
pred=predict(all_theta,X)
accuracy=np.mean(pred.flatten()==y.flatten())*100
print('Training Set Accuracy: {:.2f}%'.format(accuracy))

for i in range(num_labels):
    plt.plot(all_J_history[i], label=f'Class {i}')
plt.xlabel('Iterations')
plt.ylabel('Cost J')
plt.title('Cost over iterations for each classifier')
plt.legend()
plt.show()