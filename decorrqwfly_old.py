#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:07:48 2021

@author: nisarg
"""

import numpy as np
import sys
import numpy.linalg as lin
import math
import time
import matplotlib.pyplot as plt

pi = np.pi
cos = np.cos
sin = np.sin

#time registered from the internal clock
start = time.perf_counter()
np.set_printoptions(precision =32, threshold = np.inf, suppress = True)

##Parameter inputs

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = dtsymb*0.001 
#steps = 25*L*2*dtsymb//32 #(steps = 320000/100)

Lambda = int(sys.argv[3])
Mu =  int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

base_ = 0

if dtsymb == 1: dtstr = '1emin3'; steps = 2560
if dtsymb == 2: dtstr = '2emin3'; steps= 1280
if dtsymb == 5: dtstr = '5emin3' 

#list of the files/datasets imported for calculation

if Lambda == 1 and Mu == 0:
	param = 'qwhsbg' #'singleprec_xphsbg'
if Lambda == 0 and Mu == 1:
	param = 'qwdrvn' #'singleprec_drvn'
if Lambda ==1 and Mu==1:
        param = 'qwa2b0'; #base_ = base_ + 50000

filenum = np.arange(begin + base_,end+ base_)

def obtainspinsnew(steps):  
    #count at every 100th dt_step (100*dt = 0.005*100)
    r = int(1./dt)
    Dxt = np.ones((2*steps+1, L), dtype=np.longdouble)
    #Dnewxt = np.empty((steps+1,L))
    #t = step*r
    for j in filenum:
    	Sp_aj = np.loadtxt('./{}/{}/spin_a_{}.dat'.format(param, dtstr, str(j)))
    	Sp_bj = np.loadtxt('./{}/{}/spin_b_{}.dat'.format(param, dtstr, str(j)))
    	Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
    	Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/len(filenum)
    	
    Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
    return Dnewxt 

def savedecorr():
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    Dnewxt = obtainspinsnew(steps)
    
    f = open('./Dxt_storage/Dxt_{}_{}_dt_{}_sample_{}to{}.npy'.format(L,param, dtstr,base_ + begin,base_ + end), 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()
 
    
print(savedecorr())

stop = time.perf_counter()
print(stop-start)


