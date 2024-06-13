#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:38:50 2022

@author: nisarg
"""
import numpy as np
import sys
import numpy.linalg as lin
import math
import time
import matplotlib.pyplot as plt

start = time.perf_counter()
np.set_printoptions(precision =32, threshold = np.inf, suppress = True)

#print('for dt = 0.001, enter 1;    dt = 0.002, enter 2;    dt = 0.005, enter 5')

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])

if (dtsymb == 1):
	dt = 0.001; dtstr = '1emin3'
elif (dtsymb == 2):
	dt = 0.002; dtstr = '2emin3'
elif (dtsymb == 4):
        dt = 0.004; dtstr = '4emin3'
elif (dtsymb == 5):
        dt = 0.005; dtstr = '5emin3'

steps = 1280 #int(L*0.04/dt)//32  #was int(L*0.05/dt)//32 earlier
#these are half the total number of steps since we need only semi-cone for our analysis

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4]) #0.99 #float(sys.argv[1])
begin = int(sys.argv[5])
end = int(sys.argv[6])
filenum =  np.arange(begin,end) 

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss)

#Check plot for steps = 3200, 6400, 12800, ...
if Lambda == 1 and Mu ==0:
    param = 'qwhsbg'
if Mu == 1 and Lambda ==0:
    param = 'qwdrvn'


epstr = f'1emin{epss}'

#binary search to find the first instance of a possible duplicate value in a sorted array(boolean array mask_Dth for our purpose)
def first(arr,n): #, low, high, x, n) :   
    # if true, then return i
    for xi in range(0,n):
        if (arr[xi] == 1): return xi
    #return -1

threshfactor = int(sys.argv[8]) #100 for all purposes
#epspower: has to be either 1 or 2
#epspwr = int(sys.argv[8])
epstr = 'epsq'

path = dtstr + '/{}/times{}'.format(epstr,threshfactor) #{}.format(dtstr)
#if threshfactor == 100: path = dtstr + '/{}/times100'.format(epstr) #{}.format(dtstr)
#if threshfactor == 25: path = dtstr + '/{}/times25'.format(epstr) 

def obtainspins_thresholdtime(steps, Delta):  
    
    #threshold: the minimum value at Decorrelator front to detect
    #Dxt = np.ones(2*steps+1, L)
    
    #the initial perturbation
    ep = epsilon**2
    #the minimum value at Decorrelator front to test for
    threshold = threshfactor*ep
    timeregister_Dth = np.zeros((len(filenum), 31*L//64), dtype=np.longdouble)
    
    for j,fj in enumerate(filenum):
        Dxt = np.ones((2*steps+1, L), dtype=np.longdouble)
        Sp_aj = np.loadtxt('./{}/{}/spin_a_{}.dat'.format( param, dtstr, str(fj)))[0:3*L*(2*steps+1)]
        Sp_bj = np.loadtxt('./{}/{}/spin_b_{}.dat'.format( param, dtstr, str(fj)))[0:3*L*(2*steps+1)]
        Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
        
        #taking dot product of the spin copies at each corresponding x,t
        Dxt  -= np.sum(Sp_a*Sp_b,axis=2)
        
        Dxt_midpoint = Dxt.take(indices = range(0, 31*L//64), axis = 1)
        #convert the first arrival instance of Dxt at site x to a boolean array;
        #taking the transpose so that shape changes to (L, steps//r)
        Dxt_Tr = Dxt_midpoint.T
        mask_Dth = (Dxt_Tr >= threshold)
        Mask_Dth = mask_Dth*1
        print(np.shape(Mask_Dth))
        #Mask_Dth =  Mask_Dth[~np.isnan(Mask_Dth)]      
        #~mask_Tr = mask_Dth.T
        
        for xi in np.arange(0, 31*L//64):
            timeregister_Dth[j-1,xi] = first(Mask_Dth[xi],2*steps) 
            if np.any(Mask_Dth[xi]) != 1: timeregister_Dth[j-1,xi] = 4000
            # 1's are not present in the array
            #return -1
            '''timeregister_Dth[j-1, xi] = first(Mask_Dth[xi],0,2*steps-1,1,2*steps)'''
        print('shape of timeregister array: ', np.shape(timeregister_Dth))
        timereg_trucheck = np.all(np.array([~np.isnan(timeregister_Dth)])) #checks if there is even a single NaN in the masked Dth array
        file1 = open("nanornot_{}_{}.txt".format(param,threshfactor), "a+")
        #writing newline character
        file1.write("\n")
        print('timereg_np_all:',  timereg_trucheck)
        nancheck = str(begin) + ' ' +  str(timereg_trucheck)
        file1.write(nancheck)
        file1.close()
        #print('shape after removing NaNs of the timeregister ', np.shape(timeregister_Dth))
        #print(timeregister_Dth)
    #avg_time = np.sum(timeregister_Dth, axis = 0)/len(filenum) 
    #print (Dxt_midpoint, np.shape(Dxt_midpoint))
    #print (np.shape(mask_Dth), np.shape(mask_Tr))
    
    print('element Dxt(201,2) from last config : ', Dxt[200,1])
    print(timeregister_Dth)
    return timeregister_Dth #, avg_time


def store_arrivaltime():
    sites = np.arange(0,31*L//64)
    timeregister = obtainspins_thresholdtime(steps,Mu)
    #std_t = np.std(timeregister, axis = 0)/np.sqrt(len(avgtime))
    print('shape of timeregsiter: ', np.shape(timeregister))
    f = open('./timeregister/{}/timeregister_{}_{}_{}to{}confg.npy'.format(path, L,param, begin, end), 'wb')
    np.save(f, timeregister)
    f.close()
    print('shape of timeregister: ', np.shape(timeregister)); print('element (1,2) of timeregister: ', timeregister[0,1])
    """
    centraltime = np.zeros((len(filenum),15*L//32))
    for j in filenum:
    	centraltime[j-1] = timeregister[j-1] - avgtime"""
    
      

print(store_arrivaltime())


stop = time.perf_counter()
print(stop-start)

