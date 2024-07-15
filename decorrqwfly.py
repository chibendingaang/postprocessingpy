#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
inv_dtfact = int(1/(0.1*dtsymb))

#steps = 25*L*2*dtsymb//32 #(steps = 320000/100)

Lambda, Mu = map(float,sys.argv[3:5])
begin = int(sys.argv[5])
end = int(sys.argv[6])

base_ = 0
steps = 1280

if dtsymb == 1: dtstr = '1emin3'; steps = min(steps,1024)
if dtsymb == 2: dtstr = '2emin3'; steps = min(512,steps) #400
if dtsymb == 5: dtstr = '5emin3' 

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr)
#segregate between one and two digits after pt

#list of the files/datasets imported for calculation

if Lambda == 1 and Mu == 0:
	param = 'qwhsbg' #'singleprec_xphsbg'
elif Lambda == 0 and Mu == 1:
	param = 'qwdrvn' #'singleprec_drvn'
elif Lambda ==1 and Mu==1:
    param = 'qwa2b0'; #base_ = base_ + 50000
else:
    param = 'xpa2b0'

filenum = np.arange(begin + base_,end+ base_)

def obtainspinsnew(steps):  
    #count at every 100th dt_step (100*dt = 0.005*100)
    r = int(1./dt)
    Dxt = np.ones((inv_dtfact*steps+1, L), dtype=np.longdouble)
    #Dnewxt = np.empty((steps+1,L))
    #t = step*r
    if param == 'xpa2b0': 
        filepath = f'./{param}/L{L}/alpha_{alphastr}/{dtstr}'
    else:
        if L==1024: filepath = f'./L{L}/eps_min4/{param}'
        else: filepath = f'./{param}/{dtstr}'
    for j in filenum:
        Sp_aj = np.loadtxt(f'{filepath}/spin_a_{str(j)}.dat')
        Sp_bj = np.loadtxt(f'{filepath}/spin_b_{str(j)}.dat')
        Sp_a = np.reshape(Sp_aj, (inv_dtfact*steps+1,L,3)); 
        Sp_b = np.reshape(Sp_bj, (inv_dtfact*steps+1,L,3))           
        Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/len(filenum)
        """
        why the 2*T+1 reshape?
        this implicitly assumes that the smallest discrete unit of time is 0.005
        it will lead to incorrect t-axis if another dt_ value is used
        try 5*T+1 if dt_ == 0.002 and every 100 step is stored in the array.
        """ 	
    Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
    print(Dnewxt.shape)
    return Dnewxt 

def savedecorr():
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    Dnewxt = obtainspinsnew(steps)
    print(Dnewxt.shape, Dnewxt[::500], Dnewxt.shape)
    
    if L==1024: f = open('./Dxt_storage/L1024/Dxt_{}_{}_dt_{}_sample_{}to{}.npy'.format(L,param, dtstr,base_ + begin,base_ + end), 'wb') #Mu, epstr
    else: f = open('./Dxt_storage/alpha_ne_pm1/Dxt_{}_{}_{}_dt_{}_sample_{}to{}.npy'.format(L,param, alphastr, dtstr,base_ + begin,base_ + end), 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()
  
#print(savedecorr())
savedecorr()

stop = time.perf_counter()
print(stop-start)
