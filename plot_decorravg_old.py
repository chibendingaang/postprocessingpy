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
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

#int(L*0.05/dt)//32 #25*L//32
Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

threshfactor = 100

interval = 1

if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'

if Lambda == 1 and Mu == 1:
    param = 'qwa2b0'
    #if dt==0.002: steps = 1280
    #elif dt==0.005:  steps = 640
    #elif dt==0.001:  steps = 2560

dtstr = f'{dtsymb}emin3'

epss = int(sys.argv[7])

epsilon = 10**(-1.0*epss) #0.00001

epstr = dict()
epstr[epss] = f'emin{epss}'

if param == 'qwa2b0':
    path = f'Dxt_storage/{param}' 
    #{epstr[epss]}_{param}'
else:
    path = f'Dxt_storage/{epstr[epss]}_{param}'
filerepo = f'./{path}/{param}_dxt_{dtstr}_{epstr[epss]}_out.txt'

f = open(filerepo)
filename = np.array([x for x in f.read().splitlines()])[begin:end]
f.close()
#print(filename)

#steps = 1600*L//1024//dtsymb #it's either 1280 or 1600 as prefactor
steps = 1280*L//1024//dtsymb
Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1


for k, fk  in enumerate(filename):
	Dxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True);
	#print(Dxtk); 
	Dxtavg += Dxtk[:2*steps+1]

Dxtavg = Dxtavg/len(filename)
#D_th = 100*epsilon**2		#not needed 

t_,x_ = Dxtavg.shape
T = 0.2*np.arange(0,t_)
X = -0.5*x_ +  np.arange(0,x_)
#Dnewxt = obtainspins(steps*int(1./dt))
#vel_inst = np.empty(steps*int(1./dt)) 		#not needed


fig, ax = plt.subplots(figsize= (9, 7.5)); #plt.figure()
#fig.colorbar(img, orientation = 'horizontal')

img = ax.pcolormesh(X,T, Dxtavg[:-1,:-1], cmap = 'seismic',vmin =  0, vmax = 1.0);
ax.set_xlabel('x')
ax.set_ylabel(r'$t$')
ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu),fontsize=16) #$D(x,t)$ heat map; $t$ v/s $x$;

#plt.colorbar()
fig.colorbar(img, ax=ax)

plt.savefig('./plots/Dxt_L{}_Lambda_{}_Mu_{}_{}_dt_{}_{}confg.png'.format(L,Lambda,Mu, epstr[epss], dtstr, len(filename)))
#plt.show()

"""
for t in np.arange(0,steps//2+1, 20):
    l = int(4*t)
    ti = t
    logDxt = np.zeros(l)
    counter = 0
    
    for x in range(20,l,20):
        logDxt = np.log(Dxtavg[int(ti),L//2:L//2+l]/epsilon**2)
        f = open('./logDxt/logDxtbyepssq_L{}_t{}_lambda_{}_mu_{}_eps_1emin3_{}config.npy'.format(L,t,Lambda, Mu, len(filename)), 'wb')
        np.save(f, logDxt)
        f.close()
    #print ('t,       log(Dxt/eps*eps)/l' )
    #print()
    #print (t, logDxt/l)
"""
