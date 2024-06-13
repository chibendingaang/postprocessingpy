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

start = time.perf_counter()

L = int(sys.argv[1])
#dt = 0.002 #float(sys.argv[2]) #0.01
dtsymb =  int(sys.argv[2])
dt = 0.001*dtsymb
steps = int(0.05/dt)*L//32 #(steps = 320000/100)

Lambda = int(sys.argv[3])
Mu =  int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

base_ = 0

if dtsymb == 5: dtstr = '5emin3'
if dtsymb == 2: dtstr = '2emin3'
if dtsymb == 1: dtstr = '1emin3'

#if epss == 3: epstr = '1emin3'
#if epss == 5: epstr = '1emin5'
#if epss == 6: epstr = '1emin6'
#list of the files/datasets imported for calculation
filenum = np.arange(begin,end)


if Lambda == 1 and Mu == 0:
	param = 'hsbg' #'singleprec_hsbg'
if Lambda == 0 and Mu == 1:
	param = 'drvn'


def obtainspinsnew(steps):  
    #count at every 100th dt_step (100*dt = 0.005*100)
    r = int(1./dt)
    Dxt = np.ones((2*steps+1, L))
    #Dnewxt = np.empty((steps+1,L))
    #t = step*r
    for j in filenum:
        Sp_aj = np.loadtxt('./{}/{}/spin_a_{}.dat'.format(param, dtstr, str(j)))
        Sp_bj = np.loadtxt('./{}/{}/spin_b_{}.dat'.format(param, dtstr, str(j)))
        Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))
        Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/len(filenum)
    print ("shape of spin_a_333 file ", np.shape(Sp_aj)	)
    Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
    return Dnewxt 

#obtainspins(steps, Delta)
#if epss == 3: base_ = 0 #19800 usually
#if epss == 5: base_ = 5000
#obtainspinsnew(steps)

def savenplotdecorr():
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    #eps = 10**(-1*epss)
    Dnewxt = obtainspinsnew(steps)
    
    f = open('./Dxt_storage/Dxt_{}_lamda_{}_mu_{}_dt_{}_sample_{}to{}.npy'.format(L,Lambda,Mu, dtstr, base_ + begin,base_ + end), 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()
 
    #Dnewxt = obtainspins(steps, Mu)
    plt.figure(figsize= (9, 7.5))
    #fig.colorbar(img, orientation = 'horizontal')

    plt.pcolormesh(Dnewxt[:], cmap = 'seismic',vmin =  0, vmax = 1.0);
    plt.xlabel('x')
    plt.ylabel(r'$t$')
    plt.title(r'$D(x,t)$ heat map; $t$ v/s $x$; $\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu))
    plt.colorbar()
    plt.savefig('./plots/Dxt_L{}_Lambda_{}_Mu_{}_eps_1emin3_dt_{}_{}confg.png'.format(L,Lambda,Mu,dtstr,len(filenum)))
    #plt.show()
#sample filename:    Dxt_L128_Lambda_1_Mu_0_eps_1emin3_dt_1emin3_200confg.png 
savenplotdecorr()


stop = time.perf_counter()
print(stop-start)


