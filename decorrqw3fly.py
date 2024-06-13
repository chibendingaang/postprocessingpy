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
dt = dtsymb*0.001 #float(sys.argv[2]) #0.01
#steps = 1280 #25*L*2*dtsymb//32 #(steps = 320000/100)

Lambda = int(sys.argv[3])
Mu =  int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

base_ = 0

if dtsymb == 1: dtstr = '1emin3'; steps = 3200
if dtsymb == 2: dtstr = '2emin3'; steps = 1280
if dtsymb == 5: dtstr = '5emin3'; steps = 640

#if epss == 3: epstr = '1emin3'

if Lambda == 1 and Mu == 0:
	param = 'xphsbg' #'singleprec_xphsbg'
if Lambda == 0 and Mu == 1:
	param = 'xpdrvn' #'singleprec_drvn'
if Lambda ==1 and Mu==1:
        param = 'qwa2b0'; # base_ = base_ + 50000

#list of the files/datasets imported for calculation
filenum = np.arange(begin + base_,end+ base_)

def obtainspinsnew(steps):  
    #count at every 100th dt_step (100*dt = 0.005*100)
    r = int(1./dt)
    Dxt = np.ones((2*steps+1, L), dtype=np.longdouble)

    for j in filenum:
    	Sp_aj = np.loadtxt('./{}/{}/spin_a_{}.dat'.format(param, dtstr, str(j)))
    	Sp_bj = np.loadtxt('./{}/{}/spin_b_{}.dat'.format(param, dtstr, str(j)))
    	Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
    	Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/len(filenum)
    	
    Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
    return Dnewxt 


filename = './Dxt_storage/Dxt_{}_{}_dt_{}_sample_{}to{}.npy'.format(L,param, dtstr,base_ + begin,base_ + end)

def savedecorr():
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    Dnewxt = obtainspinsnew(steps)
    
    f = open(filename, 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()
 
#    plt.figure(figsize= (9, 7.5))
#    #fig.colorbar(img, orientation = 'horizontal')
#
#    plt.pcolormesh(Dnewxt[:], cmap = 'seismic',vmin =  0, vmax = 1.0);
#    plt.xlabel('x')
#    plt.ylabel(r'$t$')
#    plt.title(r'$D(x,t)$ heat map; $t$ v/s $x$; $\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu))
#    plt.colorbar()
#    plt.savefig('./plots/D_xt_L{}_Lambda_{}_Mu_{}_{}confg_{}steps.png'.format(L,Lambda,Mu,len(filenum), steps))
#    plt.show()
    
print(savedecorr())

stop = time.perf_counter()
print(stop-start)


