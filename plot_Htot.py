#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
#a potential source of error could be at line#55 if the file under inspection spin_a_[123].dat is
#absent

import sys
import numpy as np
import matplotlib.pyplot as plt

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

#steps = 640  #, but the first 1200-1400 odd files have 640 steps only
#int(L*0.05/dt)//32 

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

np.set_printoptions(threshold=2561)

if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'
if Lambda == 1 and Mu == 1: param = 'qwa2b0'

dtstr = f'{dtsymb}emin3'    
#steps = (1280*L//1024)/dtsymb
#if dt==0.002: dtstr = '2emin3'; steps = 1280

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 

epstr = dict()
epstr[3] =  'emin3'
epstr[4] =  'emin4' 
epstr[6] =  'emin6'
epstr[8] =  'emin8'

Cxtpath = 'Cxt_storage/L{}'.format(L)
energypath = f'H_total_storage/L{L}'

def opencorr(energ_):
    """
    retrieves ~stores the output from obtainspinsnew into a file at given path
    shape of the arrays: Cxtavg -> (2t+1,L); Sxtloc -> (2t+1,L,3); Hxtavg -> (2t+1,)
    """
    path= f'./L{L}/eps_min3/{param}'
    Sp_aj = np.loadtxt(f'{path}/spin_a_{str(end//2)}.dat')
    steps = int((Sp_aj.size/(3*L)))
    Hxtavg = np.zeros((16*(steps//640)+1,)) #2*steps+1
    
    filerepo = f'./{energypath}/{param}_{energ_}_{dtstr}_{epstr[epss]}_out.txt'#'splitetacorr{}_{}_{}.txt'.format(dtstr,param, epstr[epss])
    g = open(filerepo)
    filename2 = np.array([x for x in g.read().splitlines()])[begin:end]
    g.close()
    #print(filename2);
    
    # Sxtloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy
    
    for k, fk  in enumerate(filename2):
        Hxtk = np.load('./{}/{}'.format(energypath, fk), allow_pickle=True)[:16*(steps//640)+1]
        Hxtavg += Hxtk
    
    print('shape of H_tavg: ', Hxtavg.shape)
   
    return filename2, Hxtavg 

#filename11, filename12, Cxtavg, Sxtavg, Sx0t0 = opencorr()


energ_ = {}

#t_ = np.arange(Hetaxtavg.size)
#print(t_.shape)


def plot_Htot():
    if param=='qwhsbg':
        energ_ = 'htot'; labl = r'$H(x,t)$'; energy = 'Energy'
        filename, Hxtavg = opencorr(energ_)
        Hxtavg = Hxtavg/len(filename)
    if param=='qwdrvn':
        energ_ = 'hetatot'; labl = r'$ \tilde{H}(x,t)$'; energy = 'Pseudo-Energy'
        filename, Hxtavg = opencorr(energ_)
        Hxtavg = Hxtavg/len(filename)
        print('Ht_eta_avg = ', Hxtavg)
        #Hetacorr,  = 'Hetaxt', 'QHetaxt'
    
    fig, ax1 = plt.subplots(figsize=(10,12))
    fig.suptitle(f'{energy} correlation', fontsize=18, fontweight='bold')
    
    ax1.plot(Hxtavg)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(labl)
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    plt.savefig('./plots/{}_vs_t_L{}_lambda_{}_mu_{}_{}_{}config.png'.format(energ_,L,Lambda, Mu,epstr[epss],len(filename)))
    #plt.show()

plot_Htot()
