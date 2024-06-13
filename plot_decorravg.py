#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use('matplotlibrc')

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

#int(L*0.05/dt)//32 #25*L//32
Lambda, Mu, begin, end, epss = map(int,sys.argv[3:8])

threshfactor = 100
interval = 1 #interval is same as fine_res

dtstr = f'{dtsymb}emin3'
epsilon = 10**(-1.0*epss) #0.00001
epstr = dict()
epstr[epss] = f'emin{epss}'

if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'

if Lambda == 1 and Mu == 1: 
    param = 'qwa2b0'
    path = f'Dxt_storage/{param}' 
    #{epstr[epss]}_{param}'
else:
    path = f'Dxt_storage/{epstr[epss]}_{param}'

#files where Dxt_ arrays are stored at
filerepo = f'./{path}/{param}_dxt_{dtstr}_{epstr[epss]}_out.txt'
f = open(filerepo)
Dxt_files = np.array([dxtarr for dxtarr in f.read().splitlines()])[begin:end]
f.close()
print(Dxt_files)
configs = Dxt_files.shape[0]
print('# configs = ', configs)

steps = 1280*int(L/1024)//dtsymb #standard number of steps that have been input for the simulation runs
#what if L = 512? avoid integer division unless necessary
#either 1280 or 1600 as prefactor
Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1


for k, fk  in enumerate(Dxt_files):
    if dtsymb==1: 
        if param =='qwhsbg': Dxtk = np.load(f'./{path}/201M_1emin3/{fk}', allow_pickle=True);
        elif param =='qwdrvn': 
            if k<1296: Dxtk = np.load(f'./{path}/201M_1emin3/{fk}', allow_pickle=True);
            else: Dxtk = np.load(f'./{path}/161M_1emin3/{fk}', allow_pickle=True);
    if dtsymb==2:
        if param == 'qwdrvn': Dxtk = np.load(f'./{path}/101M_2emin3/{fk}', allow_pickle=True);
        elif param == 'qwhsbg': Dxtk = np.load(f'./{path}/81M_2emin3/{fk}', allow_pickle=True);
    Dxtavg += Dxtk[:2*steps+1]
    # why 2*steps + 1: dtsymb = 2*10**(-3); 
    # it is stored every 100 steps in dynamics --> 0.2; 5 time steps per unit time retrieved

Dxtavg = Dxtavg/len(Dxt_files)
#D_th = 100*epsilon**2		#not needed 

t_,x_ = Dxtavg.shape
T = 0.2*np.arange(0,t_)
X = -0.5*x_ +  np.arange(0,x_)
#Dnewxt = obtainspins(steps*int(1./dt))
#vel_inst = np.empty(steps*int(1./dt)) 		#not needed



fig, ax = plt.subplots(figsize= (9,7)); #plt.figure()
#fig.colorbar(img, orientation = 'horizontal')

img = ax.pcolormesh(X,T, Dxtavg[:-1,:-1], cmap = 'seismic',vmin =  0, vmax = 1.0);
ax.set_xlabel(r'$\mathbf{x}$')
ax.set_ylabel(r'$\mathbf{t}$')
#ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu),fontsize=16) #$D(x,t)$ heat map; $t$ v/s $x$;
#xticks = ticker.MaxNLocator(7)
xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
ax.set_xticks(xticks)

#plt.colorbar()
fig.colorbar(img, ax=ax,shrink=0.8 ) #, location='left')
fig.tight_layout()
plt.savefig(f'./plots/Dxt_L{L}_{param}_{epstr[epss]}_dt_{dtstr}_{configs}configs.png')
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
