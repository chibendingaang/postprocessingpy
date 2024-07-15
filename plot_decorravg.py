#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
Lambda, Mu = map(float, sys.argv[3:5])
begin, end, epss = map(int,sys.argv[5:8])

threshfactor = 100
interval = 1 #interval is same as fine_res
param = 'xpa2b0'; #'xpdrvn'; 'xphsbg'

inv_dtfact = int(1/(0.1*dtsymb))
dtstr = f'{dtsymb}emin3'
epsilon = 10**(-1.0*epss) #0.00001

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr) 

epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}

if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'

if Lambda == 1 and Mu == 1: 
    param = 'qwa2b0'
    path = f'Dxt_storage/{param}' 
    #{epstr[epss]}_{param}'
else:
    path = f'Dxt_storage/alpha_ne_pm1' #{epstr[epss]}_{param}'

#files where Dxt_ arrays are stored at
filerepo = f'./{path}/{param}_dxt_{dtstr}_{alphastr}_out.txt'
f = open(filerepo)
Dxt_files = np.array([dxtarr for dxtarr in f.read().splitlines()])[begin:end]
f.close()
print(Dxt_files[::10])
configs = Dxt_files.shape[0]

steps = int(1280*L/1024)//dtsymb 
#standard number of steps that have been input for the simulation runs
#either 1280 or 1600 as prefactor

if L == 512: 
    if dtsymb ==2: steps = 512 #min(512, steps)
    if dtsymb == 5: steps = 800
Dxtavg_ = np.zeros((inv_dtfact*steps+1,L)) #2*steps+1
print('Dxtavg_.shape: ', Dxtavg_.shape)
print('steps : ', steps)

for k, fk  in enumerate(Dxt_files):
    if dtsymb==1: 
        if param =='qwhsbg': Dxtk = np.load(f'./{path}/201M_1emin3/{fk}', allow_pickle=True);
        elif param =='qwdrvn': 
            if k<1296: Dxtk = np.load(f'./{path}/201M_1emin3/{fk}', allow_pickle=True);
            else: Dxtk = np.load(f'./{path}/161M_1emin3/{fk}', allow_pickle=True);
    if dtsymb==2:
        if param == 'qwdrvn': Dxtk = np.load(f'./{path}/101M_2emin3/{fk}', allow_pickle=True);
        elif param == 'qwhsbg': Dxtk = np.load(f'./{path}/81M_2emin3/{fk}', allow_pickle=True);
        else: Dxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
    if dtsymb==5:
        Dxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
        
    Dxtavg_ += Dxtk #[:inv_dtfact*steps+1]
    # why 2*steps + 1: dtsymb = 2*10**(-3); 
    # it is stored every 100 steps in dynamics --> 0.2; 5 time steps per unit time retrieved

Dxtavg_ = Dxtavg_/len(Dxt_files)
if dtsymb != 5 and param != 'xpa2b0': Dxtavg = np.concatenate((Dxtavg_[:, L//2:], Dxtavg_[:,0:L//2]), axis=1) 
else: Dxtavg = Dxtavg_

print(Dxtavg[::100,:])
print(Dxtavg.shape)
#D_th = 100*epsilon**2		#not needed 

t_,x_ = Dxtavg.shape
T = np.arange(0,t_) #why the factor of 0.2?
X = -0.5*x_ +  np.arange(0,x_)
#Dnewxt = obtainspins(steps*int(1./dt))
#vel_inst = np.empty(steps*int(1./dt)) 		#not needed



fig, ax = plt.subplots(figsize= (9,7)); #plt.figure()
#fig.colorbar(img, orientation = 'horizontal')

img = ax.pcolormesh(X,T, Dxtavg[:-1,:-1], cmap = 'seismic',vmin =  0, vmax = 1.0);
ax.set_xlabel(r'$\mathbf{x}$')
ax.set_ylabel(r'$\mathbf{t}$')
ax.set_ylim(0, 400)
#ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu),fontsize=16) #$D(x,t)$ heat map; $t$ v/s $x$;
#xticks = ticker.MaxNLocator(7)
#xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
#ax.set_xticks(xticks)

#plt.colorbar()
fig.colorbar(img, ax=ax,shrink=0.8 ) #, location='left')
fig.tight_layout()
plt.savefig(f'./plots/Dxt_L{L}_{param}_{alphastr}_dt_{dtstr}_{configs}configs.png')
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
