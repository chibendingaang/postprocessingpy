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
dtstr = str(dtsymb) + 'emin3'

steps = int(1280*L/1024/dtsymb)
#1280 #int(L*0.04/dt)//32  #was int(L*0.05/dt)//32 earlier
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
    param = 'qphsbg_NNN'
if Mu == 1 and Lambda ==0:
    param = 'qpdrvn_NNN'

epstr = f'emin{epss}'


#binary search to find the first instance of a possible duplicate value in a sorted array(boolean array mask_Dth for our purpose)

def first(arr,n): #, low, high, x, n) :   
    # if true, then return i
    for xi in range(0,n):
        if arr[xi]: return xi
        # since arr is already a boolean mask, 
        # arr[xi]== 1 is not explicitly stated
    #return -1

threshfactor = 100 #int(sys.argv[8]) #100 for all purposes
#epspower: has to be either 1 or 2
epspwr = 2 #int(sys.argv[9])

epsdict = dict()
epsdict[1] = 'epss'
epsdict[2] = 'epsq'

savepath = f'./timeregister' 
#/{dtstr}/{epsdict[epspwr]}/times{threshfactor}' #{}.format(dtstr)
# keep it simple!

#if threshfactor == 100: path = dtstr + '/{}/times100'.format(epstr) #{}.format(dtstr)
J2, J1 = map(float,sys.argv[8:10])
#J1 = bool(J2)^1
print('J1, J2: ', J1, J2)

interval = 1 #interval is same as fine_res
inv_dtfact = int(1/(0.1*dtsymb))
dtstr = f'{dtsymb}emin3'
epsilon = 10**(-1.0*epss) #0.00001

J21_ratio = J2/J1
def J21_lamdeci(J21_ratio): 
    if (int(1000*(J21_ratio%1))<10):
        return '00'+str(int(1000*(J21_ratio%1)))
    elif (int(1000*(J21_ratio%1)) < 100):
        return '0' + str(int(1000*(J21_ratio%1)))
    else: 
        return str(int(1000*(J21_ratio%1)))
J21_strdeci = J21_lamdeci(J21_ratio)

if J1 ==0 and J2==0: J1J2comb='invalidinteraction'
elif J1 == 0: J1J2comb = 'J2only'
elif J2 ==0: J1J2comb = 'J1only'
else: J1J2comb = 'J2byJ1_0pt' + J21_strdeci

filepath =  f'./{param}/L{L}/{dtstr}/{J1J2comb}'


def getstepcount(filepath):
    Dxt_j = np.loadtxt(f'{filepath}/Dxt_{str(11032)}.dat', dtype=np.float128) 
    # 32032, 11032: common existing filenums
    # pick relevant filenum which exists
    print('Dxt_j shape ', Dxt_j.shape)
    steps = int(Dxt_j.shape[0]/(L))
    print(steps)
    return steps


steps = getstepcount(filepath)

with open(f'{filepath}/outdxt.txt', 'r') as Dxt_repo:
    Dxt_files = np.array([dxtarr for dxtarr in Dxt_repo.read().splitlines()])[begin:end]
    # configs = Dxt_files.shape[0]

print(Dxt_files[0::len(Dxt_files)])
    
#steps = 801i
Dxtavg_ = np.zeros((steps,L), dtype=np.float128) 
print('Dxtavg_.shape: ', Dxtavg_.shape)
print('steps : ', steps)

#for Dxtk in Dxt_files:
        
def getDxtavg(Dxt_files, Dxtavg_):
    for k, fk  in enumerate(Dxt_files):
        Dxtk = np.loadtxt(f'{filepath}/{fk}', dtype=np.float128).reshape(steps,L) #when np.load was used:, allow_pickle=True)
        Dxtavg_ += Dxtk #[:inv_dtfact*steps+1]
    return Dxtavg_/len(Dxt_files)
    # why 2*steps + 1: dtsymb = 2*10**(-3); 
    # it is stored every 100 steps in dynamics --> 0.2; 5 time steps per unit time retrieved

Dxtavg_ = getDxtavg(Dxt_files, Dxtavg_)

"""Source of all misery -- To X-shift or not to?"""

Dxtavg = Dxtavg_ #np.concatenate((Dxtavg_[:, L//2:], Dxtavg_[:,0:L//2]), axis=1) 
Dnewxt = Dxtavg[:,L//2:]
logDxt = np.log(Dnewxt/epsilon**2, dtype=np.float64)



def obtainspins_thresholdtime(steps):  
    
    #threshold: the minimum value at Decorrelator front to detect
    #Dxt = np.ones(2*steps+1, L)
    
    #the initial perturbation
    ep = epsilon**epspwr
    #the minimum value at Decorrelator front to test for
    threshold = threshfactor*ep
    timeregister_Dth = np.zeros((len(filenum), 31*L//64), dtype=np.longdouble)
    
    # we have a Dxt_files .txt file record for this 
    for j,fj in enumerate(Dxt_files):
        
        #Dxt = np.ones((2*steps+1, L), dtype=np.float128)
        #spinspath = f'./{param}/{dtstr}'
        #Sp_aj = np.loadtxt(f'{spinspath}/spin_a_{str(fj)}.dat')[0:3*L*(2*steps+1)]
        #Sp_bj = np.loadtxt(f'{spinspath}/spin_b_{str(fj)}.dat')[0:3*L*(2*steps+1)]
        #Sp_aj = np.loadtxt('./L{}/eps_min{}/{}/spin_a_{}.dat'.format( L, epss, param,  str(fj)))[0:3*L*(2*steps+1)]
        #Sp_bj = np.loadtxt('./L{}/eps_min{}/{}/spin_b_{}.dat'.format( L, epss, param,  str(fj)))[0:3*L*(2*steps+1)]
        #Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
        
        #taking dot product of the spin copies at each corresponding x,t
        #Dxt  -= np.sum(Sp_a*Sp_b,axis=2)
        
        Dxt = np.loadtxt(f'{filepath}/{fj}', dtype=np.float128).reshape(steps,L)

        Dxt_midpoint = Dxt.take(indices = range(32*L//64, 63*L//64), axis = 1)
        """
        takes elements from an array along an axis.
        convert the first arrival instance of Dxt semicone at a site x>0 to a boolean array;
        taking the transpose so that shape changes to (L, steps//r)
        """
        Dxt_Tr = Dxt_midpoint.T
        mask_Dth = (Dxt_Tr >= threshold)
        Mask_Dth = mask_Dth*1
        print(np.shape(Mask_Dth))
        #Mask_Dth =  Mask_Dth[~np.isnan(Mask_Dth)]      
        #~mask_Tr = mask_Dth.T
        
        for xi in np.arange(0, 31*L//64):
            timeregister_Dth[j-1,xi] = first(Mask_Dth[xi],steps) 
            if np.any(Mask_Dth[xi]) != 1: timeregister_Dth[j-1,xi] = 4000
            # 1's are not present in the array
            #return -1
            '''timeregister_Dth[j-1, xi] = first(Mask_Dth[xi],0,2*steps-1,1,2*steps)'''
        print('shape of timeregister array: ', np.shape(timeregister_Dth))
        timereg_trucheck = np.all(np.array([~np.isnan(timeregister_Dth)])) 
        #checks if there is even a single NaN in the masked Dth array
        
        file1 = open(f"./{savepath}/nanornot_{param}_{threshfactor}.txt", "a+")
        #writing newline character
        file1.write("\n")
        #print('timereg_np_all:',  timereg_trucheck)
        nancheck = str(begin) + ' ' +  str(timereg_trucheck)
        file1.write(nancheck)
        file1.close()
        #print('shape after removing NaNs of the timeregister ', np.shape(timeregister_Dth))
        #print(timeregister_Dth)
    #avg_time = np.sum(timeregister_Dth, axis = 0)/len(filenum) 
    #print (Dxt_midpoint, np.shape(Dxt_midpoint))
    #print (np.shape(mask_Dth), np.shape(mask_Tr))
    
    #print('element Dxt(201,2) from last config : ', Dxt[200,1])
    print(timeregister_Dth)
    sites = np.arange(0,31*L//64)
    # return timeregister_Dth #, avg_time
    # timeregister = obtainspins_thresholdtime(steps)
    
    #std_t = np.std(timeregister, axis = 0)/np.sqrt(len(avgtime))
    #print('shape of timeregsiter: ', np.shape(timeregister))
    f = open('./{}/timeregister_{}_{}_{}_dt_{}_confgnum_{}.npy'.format(savepath, L, param, J1J2comb, dtstr, fj[4:9]), 'wb')
    np.save(f, timeregister_Dth)
    f.close()
    print('shape of timeregister: ', np.shape(timeregister_Dth)); print('element (1,2) of timeregister: ', timeregister_Dth[0,1])
    """
    centraltime = np.zeros((len(filenum),15*L//32))
    for j in filenum:
    	centraltime[j-1] = timeregister[j-1] - avgtime"""
    

#print(store_arrivaltime())
obtainspins_thresholdtime(steps)

stop = time.perf_counter()
print(stop-start)

