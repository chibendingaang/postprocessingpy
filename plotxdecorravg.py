#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

L = int(sys.argv[1])
dt = 0.002
steps = 25*L//32
Lambda = int(sys.argv[2])
Mu = int(sys.argv[3])
begin = int(sys.argv[4])
end = int(sys.argv[5])
interval = 1
kksmall = np.arange(begin,end,interval)
#kk = np.concatenate((np.arange(351,401),np.arange(begin,end)))

if Lambda == 1 and Mu == 0: param = 'xphsbg'
if Lambda == 0 and Mu == 1: param = 'xpdrvn'

if dt== 0.01: dtstr = '1emin2'
if dt==0.002: dtstr = '2emin3'
if dt==0.005: dtstr = '5emin3'
if dt==0.0005: dtstr = '5emin4'
Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1

epss = 3 #int(sys.argv[6])
epsilon = 10**(-1.0*epss) #0.00001
if epss == 3: 
	epss = '1emin3'; 
	base_ = 0
if epss == 5: 
	epss = '1emin5'; 
	base_ = 5000000 #5000



for k in kksmall :
	Dxtk = np.load('./Dxt_storage/Dxt_{}_{}_dt_{}_sample_{}to{}.npy'.format(L, param, dtstr, k, k+interval), allow_pickle=True);
	#print(Dxtk); 
	Dxtavg += Dxtk
#Dxtk = np.load('Dxt_{}_lamda_{}_mu_{}_sample_11001to11060.npy'.format(L, Lambda,Mu))

#Dxtk = np.load('Dxt_{}_lamda_{}_mu_{}_sample_12451to12519.npy'.format(L, Lambda,Mu))
#Dxtavg += Dxtk


Dxtavg = Dxtavg/len(kksmall)
D_th = 100*epsilon**2

#Dnewxt = obtainspins(steps*int(1./dt))
vel_inst = np.empty(steps*int(1./dt))


for t in np.arange(0,steps+1, 10):
    l = int(4*t)
    ti = t
    logDxt = np.zeros(l)
    counter = 0
    
    for x in range(10,l,10):
        logDxt = np.log(Dxtavg[int(ti),L//2:L//2+l]/epsilon**2)
        f = open('./logDxt/logDxtbyepssq_L{}_t{}_lambda_{}_mu_{}_eps_1emin3_{}config.npy'.format(L,t,Lambda, Mu, interval*len(kksmall)), 'wb')
        np.save(f, logDxt)
        f.close()
    #print ('t,       log(Dxt/eps*eps)/l' )
    #print()
    #print (t, logDxt/l)

plt.figure(figsize= (9, 7.5))
#fig.colorbar(img, orientation = 'horizontal')

plt.pcolormesh(Dxtavg[:], cmap = 'seismic',vmin =  0, vmax = 1.0);
plt.xlabel('x')
plt.ylabel(r'$t$')
plt.title(r'$\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu)) #$D(x,t)$ heat map; $t$ v/s $x$;
plt.colorbar()
plt.savefig('./plots/Dxt_L{}_{}_dt_{}_{}confg.png'.format(L,param,dtstr, interval*len(kksmall)))
#plt.show()
