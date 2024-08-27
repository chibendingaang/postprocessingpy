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

"""Definition:
Total steps = steps*dt; 
T = (=actual_steps here) 
  = arraylen/(L*3)*inv_dtfact; 
  
  where arraylen = steps//100 + 1, 
  assuming timestamp at an interval of 100 are collected
"""
inv_dtfact = int(1/(0.1*dtsymb))

steps = 1000 #25*L*2*dtsymb//32 #(steps = 320000/100)

Lambda, Mu = map(float,sys.argv[3:5])
begin = int(sys.argv[5])
end = int(sys.argv[6])
J2 = float(sys.argv[7])
J1 = bool(J2)^1
print('J1, J2: ', J1, J2)
#J1 = 0; J2 = 1
dtstr = str(dtsymb) + 'emin3'

if J1==0 and J2==0: J1J2comb='invalidinteraction'
elif J1==0 : J1J2comb = 'J2only'
elif J2==0: J1J2comb = 'J1only'
else: J1J2comb = 'J2byJ1_0pt'+ str(int(100*J2/J1))

# this is just for calculation truncation if the data has nothing interesting beyond min(640,...)
if dtsymb == 1: steps = min(steps,1024)
if dtsymb == 2: steps = min(640,steps) #400 #640
if dtsymb == 5: steps = min(1000, steps)    #800
#


if L==514: 
    if dtsymb==2: steps = 640 
    if dtsymb==5: steps = 1000
    #1000:dt_005, #we went for more than number of steps at dt_005 #640:dt_002
#else: steps = 320 ##250 for dt_005; 320 for dt_002

if L==130:
    if dtsymb==2: steps = 100
    if dtsymb==5: steps = 200 

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr)
#segregate between one and two digits after pt

#list of the files/datasets imported for calculation

if Lambda == 1 and Mu == 0:
	param = 'xphsbg_NNN' #'singleprec_xphsbg'
elif Lambda == 0 and Mu == 1:
	param = 'xpdrvn_NNN' #'singleprec_drvn'
elif Lambda ==1 and Mu==1:
    param = 'qwa2b0'; #base_ = base_ + 50000
else:
    param = 'xpa2b0'

filenum = np.arange(begin ,end)

def getstepcount(filepath):
    Sp_aj = np.loadtxt(f'{filepath}/spin_a_{str(begin)}.dat')
    print('Sp_aj shape ', Sp_aj.shape)
    steps = int(Sp_aj.shape[0]/(3*L))
    print(steps)
    return steps

def obtainspinsnew():  
    #count at every 100th dt_step (100*dt = 0.005*100)
    #t = step*r
    
    if param == 'xpa2b0': 
        filepath = f'./{param}/L{L}/alpha_{alphastr}/{dtstr}'
    else:
        if L==1024: filepath = f'./L{L}/eps_min4/{param}'
        else: filepath = f'./{param}/L{L}/{dtstr}/{J1J2comb}'
    
    steps = getstepcount(filepath)   
    print('steps = ', steps)
    r = int(1./dt)
    Dxt = np.ones((steps, L), dtype=np.longdouble)
    #inv_dtfact*steps+1 is redundant after getstepcount() is called

    for j in filenum:
        Sp_aj = np.loadtxt(f'{filepath}/spin_a_{str(j)}.dat')
        Sp_bj = np.loadtxt(f'{filepath}/spin_b_{str(j)}.dat')
        Sp_a = np.reshape(Sp_aj, (steps,L,3)); 
        #inv_dtfact*steps+1 is redundant after getstepcount() is called
        Sp_b = np.reshape(Sp_bj, (steps,L,3))           
        Dxt  -= np.sum(Sp_a*Sp_b,axis=2)
        """
        why the 2*T+1 reshape?
        this implicitly assumes that the smallest discrete unit of time is 0.005
        it will lead to incorrect t-axis if another dt_ value is used
        try 5*T+1 if dt_ == 0.002 and every 100 step is stored in the array.
        """ 	
    '''old approach: to translate the x-axis; but now we add the perturbation at site x = L//2
       to start with. So the following is not needed.
       #Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
    '''
    
    print("Dxt.shape: ", Dxt.shape)
    return Dxt/len(filenum) 

def savedecorr():
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    Dnewxt = obtainspinsnew()
    #print(Dnewxt.shape, Dnewxt[::500], Dnewxt.shape)
    
    if L==1024: f = open('./Dxt_storage/L1024/Dxt_{}_{}_dt_{}_sample_{}to{}.npy'.format(L,param, dtstr, begin, end), 'wb') #Mu, epstr
    else: f = open('./Dxt_storage/Dxt_{}_{}_{}_dt_{}_sample_{}to{}.npy'.format(L,param, J1J2comb, dtstr, begin, end), 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()
    print('Dxt: ', Dnewxt[0:steps//2:10, :])
  
#print(savedecorr())
savedecorr()

stop = time.perf_counter()
print(stop-start)
