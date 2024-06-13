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
#always double check the number of steps - it is variable across many input arrays - 640, 1280, 1600, 2560, 3200

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = dtsymb*0.001 
#steps = 25*L*2*dtsymb//32 #(steps = 320000/100)

Lambda = int(sys.argv[3])
Mu =  int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])
epss = int(sys.argv[7])
#probable values: 3,4,6,8

base_ = 0

#if dtsymb == 1: dtstr = '1emin3'; steps = 1280 #2560 if L = 2048
#if dtsymb == 2: dtstr = '2emin3'; steps= 640
#if dtsymb == 5: dtstr = '5emin3' 
dtstr = f'{dtsymb}emin3'


#steps = int(1280*L/1024/dtsymb)
#list of the files/datasets imported for calculation

if Lambda == 1 and Mu == 0:
	param = 'qwhsbg' #'singleprec_xphsbg'
if Lambda == 0 and Mu == 1:
	param = 'qwdrvn' #'singleprec_drvn'
if Lambda ==1 and Mu==1:
        param = 'qwa2b0'; #base_ = base_ + 50000

path = f'L{L}/eps_min{epss}/{param}'
Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(begin)))
steps = int((Sp_aj.size/(3*L))) #0.1*dtsymb multiplication omitted
print('steps = ', steps)
filenum = np.arange(begin + base_,end+ base_)

def obtainspinsnew(steps):  
    #count at every 100th dt_step (100*dt = 0.005*100)
    r = int(1./dt)
    Dxt = np.ones((steps, L), dtype=np.longdouble)
    #Dnewxt = np.empty((steps+1,L))
    #t = step*r
    for j in filenum:
    	Sp_aj = np.loadtxt('./L{}/eps_min{}/{}/spin_a_{}.dat'.format(L, epss, param, str(j)))
    	Sp_bj = np.loadtxt('./L{}/eps_min{}/{}/spin_b_{}.dat'.format(L, epss, param, str(j)))
    	Sp_a = np.reshape(Sp_aj, (steps,L,3)); Sp_b = np.reshape(Sp_bj, (steps,L,3))           
    	Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/len(filenum)
    	
    Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
    return Dnewxt 
    #shape has to be (2t+1,L)

def savedecorr():
    dxtpath = 'Dxt_storage'
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    Dnewxt = obtainspinsnew(steps)
    
    f = open('./{}/L{}/Dxt_{}_{}_emin{}_dt_{}_sample_{}to{}.npy'.format(dxtpath, L, L, param, epss, dtstr,base_ + begin,base_ + end), 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()
 
print(savedecorr())

stop = time.perf_counter()
print(stop-start)


