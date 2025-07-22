import numpy as np

data=np.loadtxt('/Users/liupuchen/VsCode/PythonProject/LinearRegression-study/ex1data2.txt',delimiter=',')
X=data[:,0:2]
X=np.c_[np.ones(X.shape[0]),X]
y=data[:,2]
theta=np.linalg.inv(X.T@X)@X.T@y
print(theta)
price=[1,1650,3]@theta
print(price)