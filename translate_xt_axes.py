iimport numpy as np

np.random.seed(98)
L = 5; steps = 8
#arr = np.random.uniform(-1,1,(steps+1, L,3))
arr = np.reshape(np.arange(1, 3*L*steps+1), (steps, L, 3))
a,b,c = arr.shape
print('array shape: ', arr.shape)

# we are checking that the dimension of the first x and first y sliced elements 
# of the array is the same as the last x and the last y sliced elements 
for x in range(a+1):
    for y in range(b+1):
        K1 = arr[x:, y:, :]
        K2 = arr[:a-x, :b-y, :]
        #print(K1.shape, K2.shape)
        #print(K1, K2)
        

arr1 = np.roll(arr, -2, axis = 1)
arr2 = np.roll(arr, 2, axis = 1)
arr3 = np.roll(arr, b-3, axis = 1)

print('arrays 0,1,2,3 without and with translation ')
print('\n arr steps 0-1: \n', arr[0:2])
print('\n arr1 steps 0-1: \n', arr1[0:2])
print('\n arr2 steps 0-1: \n', arr2[0:2])
print('\n arr3 steps 0-1: \n', arr3[0:2])
