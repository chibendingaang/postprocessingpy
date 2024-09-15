#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nisarg
"""
# Quick and dirty copy of plot_decorravg.py

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use('matplotlibrc')
#from decorrfly_nnn import getstepcount

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

#int(L*0.05/dt)//32 #25*L//32
Lambda, Mu = map(float, sys.argv[3:5])
begin, end, epss = map(int,sys.argv[5:8])
J2, J1 = map(float,sys.argv[8:10])
#J1 = bool(J2)^1
print('J1, J2: ', J1, J2)

threshfactor = 100
interval = 1 #interval is same as fine_res
param = 'xpa2b0'; #'xpdrvn'; 'xphsbg'

inv_dtfact = int(1/(0.1*dtsymb))
dtstr = f'{dtsymb}emin3'
epsilon = 10**(-1.0*epss) #0.00001

if J1 ==0 and J2==0: J1J2comb='invalidinteraction'
elif J1 == 0: J1J2comb = 'J2only'
elif J2 ==0: J1J2comb = 'J1only'
else: J1J2comb = 'J2byJ1_0pt' + str(int(1000*J2/J1))

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr) 

epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}

def getpathnparam():
    if Lambda == 1 and Mu == 0: 
        param = 'qphsbg_NNN'
        path = f'Dxt_storage'
    elif Lambda == 0 and Mu == 1: 
        param = 'qpdrvn_NNN'
        path = f'Dxt_storage'
    elif Lambda == 1 and Mu == 1: 
        param = 'qwa2b0'
        path = f'Dxt_storage/{param}' 
        #{epstr[epss]}_{param}'
    else:
        param = 'xpa2b0'
        path = f'Dxt_storage/alpha_ne_pm1' #{epstr[epss]}_{param}'
    return param, path

param, path = getpathnparam()



if param == 'xpa2b0': 
    filepath = f'./{param}/L{L}/alpha_{alphastr}/{dtstr}'
else:
    if L==1024: filepath = f'./L{L}/eps_min4/{param}'
    else: filepath = f'./{param}/L{L}/{dtstr}/{J1J2comb}'


def getstepcount():
    Sp_aj = np.loadtxt(f'{filepath}/spin_a_{str(1)}.dat', dtype=np.float128) #pick relevant filenum #4002 for L=130
    steps = int(Sp_aj.shape[0]/(3*L))
    print('steps: ', steps)
    return steps

steps = getstepcount()

#files where Dxt_ arrays are stored at
filerepo = f'./{path}/{param}_dxt_{L}_{dtstr}_{J1J2comb}_out.txt' #{J1J2comb}_0pt80
#filerepo = f'./{path}/{param}_dxt_{dtstr}_{alphastr}_out.txt'
print('filerepo name, ', filerepo)

f = open(filerepo)
Dxt_files = np.array([dxtarr for dxtarr in f.read().splitlines()])[begin:end]
f.close()
print(Dxt_files[::10])
configs = Dxt_files.shape[0]

#steps = 801
Dxtavg_ = np.zeros((steps,L), dtype=np.float128) 
print('Dxtavg_.shape: ', Dxtavg_.shape)
print('steps : ', steps)

def getDxtavg(Dxt_files, Dxtavg_):
    for k, fk  in enumerate(Dxt_files):
        if dtsymb==1: 
            if param =='qwhsbg': Dxtk = np.load(f'./{path}/201M_1emin3/{fk}', allow_pickle=True);
            elif param =='qwdrvn': 
                if k<1296: Dxtk = np.load(f'./{path}/201M_1emin3/{fk}', allow_pickle=True);
                else: Dxtk = np.load(f'./{path}/161M_1emin3/{fk}', allow_pickle=True);
            Dxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
        if dtsymb==2:
            if param == 'qwdrvn': Dxtk = np.load(f'./{path}/101M_2emin3/{fk}', allow_pickle=True);
            elif param == 'qwhsbg': Dxtk = np.load(f'./{path}/81M_2emin3/{fk}', allow_pickle=True);
            Dxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
        if dtsymb==5:
            Dxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
        
        Dxtavg_ += Dxtk #[:inv_dtfact*steps+1]
    return Dxtavg_/len(Dxt_files)
    # why 2*steps + 1: dtsymb = 2*10**(-3); 
    # it is stored every 100 steps in dynamics --> 0.2; 5 time steps per unit time retrieved

Dxtavg_ = getDxtavg(Dxt_files, Dxtavg_)

"""Source of all misery -- To X-shift or not to?"""

Dxtavg = Dxtavg_ #np.concatenate((Dxtavg_[:, L//2:], Dxtavg_[:,0:L//2]), axis=1) 
if dtsymb == 5 and param == 'xpa2b0': Dxtavg = Dxtavg_

np.set_printoptions(threshold=sys.maxsize)
print(Dxtavg[0:10,1::2])
print(Dxtavg[0:10,::2])
#D_th = 100*epsilon**2		#not needed 

#####  #####
t_,x_ = Dxtavg.shape
#####  #####
#inv_dtfact was needed to reshape the raw Dxt file: however,
#the actual number of steps is dt*raw_steps in the param file

T = np.arange(0,t_) #why the factor of 0.2?
X = -0.5*x_ +  np.arange(0,x_)
#Dnewxt = obtainspins(steps*int(1./dt))
#vel_inst = np.empty(steps*int(1./dt)) 		#not needed

def plot_decorravg(X, T, Dxtavg, L, steps, param, J1J2comb, dtstr, configs):
    # First plot
    fig, ax = plt.subplots(figsize=(9, 7))  # plt.figure()
    # fig.colorbar(img, orientation='horizontal')
    
    img = ax.pcolormesh(X, dtsymb*T, Dxtavg[:-1, :-1], cmap='seismic', vmin=0, vmax=1.0)
    ax.set_xlabel(r'$\mathbf{x}$')
    ax.set_ylabel(r'$\mathbf{t}$')
    ax.set_ylim(0, dtsymb*steps*8//9)
    # ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16)  # $D(x,t)$ heat map; $t$ v/s $x$;
    # xticks = ticker.MaxNLocator(7)
    # xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
    # ax.set_xticks(xticks)
    
    # plt.colorbar()
    fig.colorbar(img, ax=ax, shrink=0.8)  # , location='left')
    fig.tight_layout()
    plt.savefig(f'./plots/Dxt_L{L}_{param}_{J1J2comb}_dt_{dtstr}_{configs}configs.png')
    # change J1J2comb with alphastr depending on what the file source is
    # plt.show()
    

    # Second plot -- of sublattices
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7))  # plt.figure()
    # fig.colorbar(img, orientation='horizontal')
    
    # due to the initial perturbation being at x==L//2, for 1600, x=800, which in python is xi=799, for x=1602, xi=800
    if L % 4 == 2:
        img1 = ax1.pcolormesh(X[1::2], T, Dxtavg[:-1, 1::2], cmap='seismic', vmin=0, vmax=1.0)
        img2 = ax2.pcolormesh(X[0::2], T, Dxtavg[:-1, 0::2], cmap='seismic', vmin=0, vmax=1.0)
    elif L % 4 == 0:
        img1 = ax1.pcolormesh(X[1::2], T, Dxtavg[:-1, ::2], cmap='seismic', vmin=0, vmax=1.0)
        img2 = ax2.pcolormesh(X[0::2], T, Dxtavg[:-1, 1::2], cmap='seismic', vmin=0, vmax=1.0)
    
    ax1.set_xlabel(r'$\mathbf{x}$')
    ax1.set_ylabel(r'$\mathbf{t}$')
    ax1.set_ylim(0, steps)
    # ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16)  # $D(x,t)$ heat map; $t$ v/s $x$;
    # xticks = ticker.MaxNLocator(7)
    # xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
    # ax.set_xticks(xticks)
    
    ax2.set_xlabel(r'$\mathbf{x}$')
    ax2.set_ylabel(r'$\mathbf{t}$')
    ax2.set_ylim(0, steps)
    # ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16)  # $D(x,t)$ heat map; $t$ v/s $x$;
    # xticks = ticker.MaxNLocator(7)
    # xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
    # ax.set_xticks(xticks)
    # plt.colorbar()
    
    #fig2.colorbar(img1, ax=[ax1, ax2], shrink=0.8)  # , location='left')
    #fig2.tight_layout()
    plt.savefig(f'./plots/Dxt_splitlattices_L{L}_{param}_{J1J2comb}_dt_{dtstr}_{configs}configs.png')
    # change J1J2comb with alphastr depending on what the file source is
    # plt.show()
    
 
plot_decorravg(X, T, Dxtavg, L, steps, param, J1J2comb, dtstr, configs)

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
