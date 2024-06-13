import numpy as np
"""
x = np.arange(60).reshape(3,4,5)
y = np.arange(24).reshape(4,3,2)
 
z = np.tensordot(x,y,axes=([1,0],[0,1]))

print(x)
print(y)
print(z)

X = np.arange(0,36000,0.01).reshape(12000, 100,3)
Y = np.random.uniform(0,36000, (12000,100,3))
print(X.shape)
print(Y.shape)
Z = np.tensordot(X,Y,axes)
"""
X = np.arange(0,3600,0.1).reshape(240, 50,3)
Y = np.random.uniform(0,3600, (240,50,3))

Z = np.inner(X,Y)
#print(Y.shape)
#Z = np.sum(X*Y, axis=2)
print(Z.shape)