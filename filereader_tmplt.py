import numpy as np
from collections import deque

filepath = 'Dxt_storage/L1024'
filename = 'outDxtmin4.txt'

f = open(f'./{filepath}/{filename}')
files_ = f.readlines()
print(len(files_))
x = []

print('len of reading array: ', len(files_))
'''
for filex in files_:
    tmp = np.loadtxt(f'./{filepath}/{filex.split()[0]}')
    x.append(filex.split()[0])
    #print(tmp.shape)
'''
f.close()

#print(x)




"""
#This whole exercise is useless
#deque being a doubly linked list, printing elements or storing/transferring/retrieving them requires slightly different 
#sets of commands from it

files = deque()
xl = len(files_og)

while xl > 0:
    files.appendleft(files_og.pop())
    xl -= 1

print(files[0])
"""
