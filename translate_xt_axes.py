import numpy as np

np.random.seed(98)
L = 8; steps = 48
arr = np.random.uniform(-1,1,(steps+1, L,3))
a,b,c = arr.shape

# we are checking that the dimension of the first x and first y sliced elements 
# of the array is the same as the last x and the last y sliced elements 
for x in range(a+1):
    for y in range(b+1):
        K1 = arr[x:, y:, :]
        K2 = arr[:a-x, :b-y, :]
        print(K1.shape, K2.shape)
