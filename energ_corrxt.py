#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:01:06 2023

@author: nisarg
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

threshfactor = 100
interval = 1

if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'
if Lambda == 1 and Mu == 1: param = 'qwa2b0'
dtstr = f'{dtsymb}emin3'    

np.set_printoptions(threshold=2561)
#length of the array

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss)
#dictionary names
epstr = dict()
epstr[3] = 'eps_min3'
epstr[4] = 'eps_min4'
epstr[6] = 'eps_min6'
epstr[8] = 'eps_min8' 

choice = int(sys.argv[8])
start  = time.perf_counter()

if choice ==0:
    path = f'./L{L}/{epstr[epss]}/{param}'
elif choice ==1:
    path = './{}/{}'.format(param,dtstr) 	#the epss = 3 for this path

"""settling the steps count discrepancy for good:
Logic for 2*steps + 1:
dt = 0.001 (or 0.002, 0.005)
stored arrays at 100 step_intervals -> (0.1)*dtsymb in time unit
so steps = (len(array)/(3*L) - 1)*0.1*dtsymb
""" 

#if (begin >8000) and (begin <8192):

#choice ==0 -> current location of data;
#choice ==1 -> some other location



def obtainspinsnew(file_j): 
    """
    collects the spin configurations a,b for a given initial condition; and finds the time-correlator for them
    step count is kept at every 1/10th of unit time; for example, dt = 0.001 r = 1000
    t = step*r
    
    input: number of steps of the dynamics
    returns: 
    Cxt- the single configuration spin/magnetization (time-)correlator
    Cnnxt- single configuraiton staggered magnetization (time-)correlator
    energ_1- system energy at time
    energ_eta1- eta-based system energy at time
    """ 

    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(begin)))
    steps = int((Sp_aj.size/(3*L))) #0.1*dtsymb multiplication omitted
    print('steps = ', steps)
    
    #r = int(1./dt)
    j = file_j
 
    #step = 1280*L//(1024*dtsymb); #steps = 2560 if L= 2048 #1280 if L = L1024 #1600 if L = 2048
    CExt = np.zeros((steps//40+1, L))
    CENxt = np.zeros((steps//40+1, L))
    print('CExt.shape: ', CExt.shape)
    
    #choose the unperturbed configuration for our purposes
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(j)))
    Sp_a = np.reshape(Sp_aj, (steps,L,3))[0:steps:40]
    #reshaped to count spin_xt at every 4th time step

    print('Sp_a.shape: ' , Sp_a.shape)
    Spnbr_a = np.roll(Sp_a,-1,axis=1) 		
    #the next list/array of 3-components along axis=1 has to be shifted up; so -1, not -3
    #np.roll also takes care of Periodic Boundary condition 
    
    """
    C(x,t) = 1/(L*T)* \Sum_n S_n(x+x',t+t') S_n(x',t')
    """
    # for example:
    # x= (np.arange(1,(2*steps+1)*L*3 +1)).reshape(2*steps+1, L, 3)
    # y = 0.5*np.ones(x.shape)
    # z = np.sum(x*y[0,0],axis=2) -> z.shape = (2*steps+1, L)
    #S_loc = Sp_a; N_loc = Sp_ngva

    #if param == 'qwhsbg':
    E_loc = (-1)*np.sum(Sp_a*Spnbr_a,axis=2) #energy at each site
    energ_1 = np.sum(E_loc, axis = 1) 	#attached the (-1) factor
    for ti,t in enumerate(range(0,steps,40)):
        #print(ti)
        for x in range(L):
            CExt[ti,x] = np.sum((E_loc*np.roll(np.roll(E_loc,-x,axis=1),-ti,axis=0))[:-ti])/(steps//40+1 - ti)  	#multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
    #array[-:ti] acts like step function: includes matrix products uptil that T-ti
    #return Cxt/L, energ_1 

    #if param == 'qwdrvn':
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a*eta
    Spnbr_ngva = Spnbr_a*eta 
    Eeta_loc = (-1)*np.sum(Sp_a*Spnbr_ngva,axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis = 1) 	#attached (-1) factor
    for ti,t in enumerate(range(0,steps,40)):
        #print(ti)
        for x in range(L):
            CENxt[ti,x] = np.sum((Eeta_loc*np.roll(np.roll(Eeta_loc,-x,axis=1),-ti,axis=0))[:-ti])/(steps//40+1 - ti) 	#multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
            #Cnnxt = np.sum(Sp_ngva*Sp_ngva[0,0], axis=2)
    return CExt/L, CENxt/L, energ_1, energ_eta1
    
    print('shapes of the arrays CExt/CENxt, energ_1/energ_eta1 respectively: ')
    print(CExt.shape, energ_1.shape)
    # check if the respective shapes are: 
    # (2t+1,L), (2t+1,)

Cxtpath = 'Cxt_series_storage/L{}'.format(L)
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)
'''
def savedecorr(file_j):
    """
    stores the output from obtainspinsnew into a file at given path
    """
    #r = int(1./dt); 
    if param == 'qwhsbg':
        Cxt, energ_1 = obtainspinsnew(steps,file_j)
    if param != 'qwdrvn':
        Cnnxt, energ_eta1 = obtainspinsnew(steps,file_j)
 
    if Cxt:
        f = open('./{}/Cxt_{}_{}_dt_{}_sample_{}to{}.npy'.format/
             (Cxtpath,L,param, dtstr,base_ + begin,base_ + end), 'wb') #Mu, epstr[epss]
        np.save(f, Cxt); f.close()    

    #filerepo = 'splitdec{}{}_{}.txt'.format(threshfactor, param, dtstr)
'''

"""
f = open(filerepo)
filename = np.array([x for x in f.read().splitlines()])[begin:end]
f.close()
#print(filename)
"""

nconf = len(range(begin,end))
#steps = 1280*L//(1024*dtsymb); #steps = 2560 if L= 2048 #1280 if L = L1024 #1600 if L = 2048

# you have already attached the shell scripting part in the python script; hence keep begin, end
# to be the range of the available files

for conf in range(begin, end):
    #if param == 'qwhsbg':
    CExt, CENxt, energ_1, energ_eta1  = obtainspinsnew(conf)
    #if param == 'qwdrvn':
    #    Cnnxt, energ_eta1 = obtainspinsnew(conf)
    #Cxt,Cnnxt,energ_1,energ_eta1, S_loc, N_loc, E_loc, Eeta_loc  = obtainspinsnew(steps,conf)
    #Cxtavg += Cxt
    
    #easier condition: if Cxt (is not None): save the qwhsbg files; if Cnnxt(is not None): save the qwdrvn files;
    #if param == 'qwhsbg':
    f = open(f'./{CExtpath}/Cxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, CExt); f.close()
    #if param == 'qwdrvn':
    f = open(f'./{CExtpath}/Cnnxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, CENxt); f.close()
    
    
    #if param == 'qwhsbg':
    #g = open(f'./{energypath}/H_tot_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_1); g.close()
    #if param == 'qwdrvn':
    #g = open(f'./{energypath}/H_tot_etat_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_eta1); g.close()
    '''
    if param != 'qwhsbg':
        f = open(f'./{CExtpath}/Sxtloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); np.save(f, S_loc); f.close()
        g = open(f'./{CExtpath}/Nxtloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); np.save(g, N_loc); g.close()
    if param != 'qwdrvn':
        g = open(f'./{energypath}/Etloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); np.save(g, E_loc); g.close()
    if param != 'qwhsbg':
        g = open(f'./{energypath}/Eetatloc_tot_etat_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); 
        np.save(g, Eeta_loc); g.close()
    '''

print('processing time = ', time.perf_counter() - start)

#Cxtavg = Cxtavg/nconf
#print('shape of Cxtavg: ', Cxtavg.shape)

