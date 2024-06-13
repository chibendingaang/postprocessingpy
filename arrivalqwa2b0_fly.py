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
	dt = 0.001; dtstr = '1emin3'; steps = 2560
elif (dtsymb == 2):
	dt = 0.002; dtstr = '2emin3'; steps = 1280
elif (dtsymb == 5):
        dt = 0.005; dtstr = '5emin3'; steps = 640

#steps = int(L*0.05/dt)//32  #(10*L//32 for dt = 0.005)
#these are half the total number of steps since we need only semi-cone for our analysis

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4]) #0.99 #float(sys.argv[1])
begin = int(sys.argv[5])
end = int(sys.argv[6])
filenum =  np.arange(begin,end) 



#Check plot for steps = 3200, 6400, 12800, ...
if Lambda == 1 and Mu ==0:
    param = 'qwhsbg'
if Mu == 1 and Lambda ==0:
    param = 'qwdrvn'
if Mu ==1 and Lambda ==1 :
    param = 'qwa2b0'


#binary search to find the first instance of a possible duplicate value in a sorted array(boolean array mask_Dth for our purpose)
def first(arr,n): #, low, high, x, n) :   
    # if true, then return i
    #find() method should work as well: xi = arr.find(1)
    for xi in range(0,n):
        if (arr[xi] == 1): return xi
    #return -1


threshfactor = int(sys.argv[7]) #100 for all purposes
#epspower: has to be either 1 or 2
epspwr = int(sys.argv[8])
#epss = 3
epsilon = 10**(-1.0*3)
#if epspwr == 1 and threshfactor == 100: path = dtstr + '/epss/times100' #{}.format(dtstr)
if epspwr == 2 and threshfactor == 100: path = dtstr + '/epsq/times100'
elif epspwr == 2 and threshfactor == 25: path = dtstr + '/epsq/times25'

def obtainspins_thresholdtime(steps, Delta):      
    #the initial perturbation
    ep = epsilon**epspwr

    #threshold: the minimum value at Decorrelator front to detect
    #Dxt = np.ones(2*steps+1, L)
    threshold = threshfactor*ep
    #timeregister_rDth = np.zeros((len(filenum), 16*L//64), dtype=np.longdouble)
    timeregister_lDth = np.zeros((len(filenum), 31*L//64), dtype=np.longdouble)
    
    for j,fj in enumerate(filenum):
        Dxt = np.ones((2*steps+1, L), dtype=np.longdouble)
        Sp_aj = np.loadtxt('./{}/{}/spin_a_{}.dat'.format( param, dtstr, str(fj)))
        Sp_bj = np.loadtxt('./{}/{}/spin_b_{}.dat'.format( param, dtstr, str(fj)))
        Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
        
        #taking dot product of the spin copies at each corresponding x,t
        Dxt  -= np.sum(Sp_a*Sp_b,axis=2)
        
        Dxt_midpoint = Dxt.take(indices = range(0, 16*L//64), axis = 1)
        #flipping back-front to treat the left cone like right bending cone for the first(arr,n) function
        Dxt_leftcone = np.flip(Dxt.take(indices = range(33*L//64, L), axis = 1), axis = 1)
        
        #taking the transpose so that shape changes to (L, steps//r)
        Dxtrcone_Tr = Dxt_midpoint.T
        Dxtlcone_Tr  = Dxt_leftcone.T
        
        #convert the first arrival instance of Dxt at site x to a boolean array;

        maskrcone_Dth = (Dxtrcone_Tr >= threshold); masklcone_Dth = (Dxtlcone_Tr <= threshold)
        Maskrcone_Dth = maskrcone_Dth*1; Masklcone_Dth = masklcone_Dth*1
        print(np.shape(Maskrcone_Dth)); print(np.shape(Masklcone_Dth))
        
        #Mask_Dth =  Mask_Dth[~np.isnan(Mask_Dth)]      
        #~mask_Tr = mask_Dth.T
        
        #for xi in np.arange(0, 16*L//64):
            #timeregister_rDth[j-1,xi] = first(Maskrcone_Dth[xi],2*steps) 
            #if np.any(Maskrcone_Dth[xi]) != 1: print('no threshold cross uptil index {}'.format(xi)) #timeregister_Dth[j-1,xi] = 4000
            
        for xi in np.arange(0, 31*L//64):
            timeregister_lDth[j-1,xi] = first(Masklcone_Dth[xi],2*steps) 
            if np.any(Masklcone_Dth[xi]) != 1: print('no threshold cross uptil index {}'.format(xi)) #timeregister_Dth[j-1,xi] = 4000
            # 1's are not present in the array
            #return -1
            '''timeregister_lDth[j-1, xi] = first(Masklcone_Dth[xi],0,2*steps-1,1,2*steps)'''
        
        print('shape of left boundary timeregister array: ', np.shape(timeregister_lDth))
        
        timereg_trucheck = np.all(np.array([~np.isnan(timeregister_lDth)])) #checks if there is even a single NaN in the masked Dth array
        file1 = open("nanornot_{}_{}.txt".format(param,threshfactor), "a+")
        #writing newline character
        file1.write("\n")
        print('timereg_np_all:',  timereg_trucheck)
        nancheck = str(begin) + ' ' +  str(timereg_trucheck)
        file1.write(nancheck)
        file1.close()
        #print('shape after removing NaNs of the timeregister ', np.shape(timeregister_Dth))
    #print (np.shape(mask_Dth), np.shape(mask_Tr))

    print('element Dxt(201,2) from last config : ', Dxt[200,1])
    return timeregister_lDth #, avg_time


def store_arrivaltime():
    #sites_r = np.arange(0,16*L//64)
    sites_l = np.arange(L//2, 63*L//64)
    timeregister_l = obtainspins_thresholdtime(steps,Mu)
    print('shape of timeregister: ', np.shape(timeregister_l))

    g = open('./timeregister/{}/timeregister_left__{}_{}_{}to{}confg.npy'.format(path, L,param, begin, end), 'wb')
    np.save(g, timeregister_l)
    g.close()
    print('shape of timeregister: ', np.shape(timeregister_l)); print('element (1,2) of timeregister: ', timeregister_l[0,1])
      

print(store_arrivaltime())


stop = time.perf_counter()
print(stop-start)

