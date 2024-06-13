#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First version on Wed Oct 13 17:46:47 2021

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
#    steps = 320
#print('steps: ', steps)

#choice ==0 -> current location of data;
#choice ==1 -> some other location
#Cxtavg = np.zeros((2*steps+1,L)) #2*steps+1

def calc_Cxt(Cxt_, steps, spin):
    for ti,t in enumerate(range(0,steps,10)):
        for x in range(L):
            Cxt_[ti,x] = np.sum(np.sum((spin*np.roll(np.roll(spin,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(steps//10+1 - ti)  	#multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
    return Cxt_

def calc_CExt(CExt_, steps, elocal):
    for ti,t in enumerate(range(0,steps,10)):
        for x in range(L):
            CExt_[ti,x] = np.sum((elocal*np.roll(np.roll(elocal,-x,axis=1),-ti,axis=0))[:-ti])/(steps//10+1 - ti)
    return CExt_

def obtaincorrxt(file_j): 
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

    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(end)))
    steps = int((Sp_aj.size/(3*L))) #0.1*dtsymb multiplication omitted
    #reshaped to count spin_xt at every 4th time step
    Sp_a = np.reshape(Sp_aj, (steps,L,3))[0:steps:10]
    print('Sp_a.shape: ' , Sp_a.shape)
    print('steps = ', steps)
    
    r = int(1./dt)
    j = file_j
    Cxt = np.zeros((steps//10+1, L))
    Cnnxt = np.zeros((steps//10+1, L))
    CExt = np.zeros((steps//10+1, L))
    CENxt = np.zeros((steps//10+1, L))
    print('Cxt.shape: ', Cxt.shape)
    print('CExt.shape: ', CExt.shape)
    

    #tryin np.roll:
    #print((Sp_a*np.roll(np.roll(Sp_a,-498,axis=1),-320,axis=0))[:-320].shape)
    Spnbr_a = np.roll(Sp_a,-1,axis=1) 		
    #the next list/array of 3-components along axis=1 has to be shifted up; so -1, not -3
    #np.roll also takes care of Periodic Boundary condition 
    
    """
    Definition:
    C(x,t) = 1/(L*(T-t))* \Sum_{x',t'} S_n(x+x',t+t') S_n(x',t')
    """
    # for example:
    # x= (np.arange(1,(2*steps+1)*L*3 +1)).reshape(2*steps+1, L, 3)
    # y = 0.5*np.ones(x.shape)
    # z = np.sum(x*y[0,0],axis=2) -> z.shape = (2*steps+1, L)
    #S_loc = Sp_a; N_loc = Sp_ngva

    #if param == 'qwhsbg':
    E_loc = (-1)*np.sum(Sp_a*Spnbr_a,axis=2) #energy at each site
    energ_1 = np.sum(E_loc, axis = 1) 	#attached the (-1) factor
    '''for ti,t in enumerate(range(0,steps,40)):
        for x in range(L):
            Cxt[ti,x] = np.sum(np.sum((Sp_a*np.roll(np.roll(Sp_a,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(steps//40+1 - ti)  	#multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
            CExt[ti,x] = np.sum((E_loc*np.roll(np.roll(E_loc,-x,axis=1),-ti,axis=0))[:-ti])/(steps//40+1 - ti)
    '''
    #array[-:ti] acts like step function: includes matrix products uptil that T-ti
    #return Cxt/L, energ_1 

    #if param == 'qwdrvn':
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a*eta
    Spnbr_ngva = Spnbr_a*eta 
    Eeta_loc = (-1)*np.sum(Sp_a*Spnbr_ngva,axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis = 1) 	#attached (-1) factor
    '''for ti,t in enumerate(range(0,steps,40)):
        for x in range(L):
            Cnnxt[ti,x] = np.sum(np.sum((Sp_ngva*np.roll(np.roll(Sp_ngva,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(steps//40+1 - ti) 	
            CENxt[ti,x] = np.sum((Eeta_loc*np.roll(np.roll(Eeta_loc,-x,axis=1),-ti,axis=0))[:-ti])/(steps//40+1 - ti)
            #Cnnxt = np.sum(Sp_ngva*Sp_ngva[0,0], axis=2)
    '''
    Cxt = calc_Cxt(Cxt,steps, Sp_a); Cnnxt = calc_Cxt(Cnnxt,steps, Sp_ngva)
    CExt = calc_CExt(CExt,steps, E_loc); CENxt = calc_CExt(CENxt,steps, Eeta_loc)
    return Cxt/L, Cnnxt/L, CExt/L, CENxt/L, energ_1, energ_eta1
    
    print('shapes of the arrays Cxt/Cnnxt, energ_1/energ_eta1 respectively: ')
    print(Cxt.shape, energ_1.shape)
    # check if the respective shapes are: 
    # (2t+1,L), (2t+1,)

Cxtpath = 'Cxt_storage/L{}'.format(L)
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)


nconf = len(range(begin,end))


for conf in range(begin, end):
    #if param == 'qwhsbg':
    Cxt, Cnnxt, CExt, CENxt, energ_1, energ_eta1  = obtaincorrxt(conf)
    #if param == 'qwdrvn':
    #    Cnnxt, energ_eta1 = obtaincorrxt(conf)
    #Cxt,Cnnxt,energ_1,energ_eta1, S_loc, N_loc, E_loc, Eeta_loc  = obtaincorrxt(steps,conf)
    #Cxtavg += Cxt
    
    #easier condition: if Cxt (is not None): save the qwhsbg files; if Cnnxt(is not None): save the qwdrvn files;
    #if param == 'qwhsbg':
    f = open(f'./{Cxtpath}/Cxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, Cxt); f.close()
    #if param == 'qwdrvn':
    f = open(f'./{Cxtpath}/Cnnxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, Cnnxt); f.close()
    
    #ideally, should've been named CExt, and CENxt for storage purposes, but we're continuing it since many files have been 
    #already saved with this name
    #h = open(f'./{CExtpath}/Cxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(h, CExt); h.close()
    #if param == 'qwdrvn':
    #h = open(f'./{CExtpath}/Cnnxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(h, CENxt); h.close()
    
    #if param == 'qwhsbg':
    #g = open(f'./{energypath}/H_tot_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_1); g.close()
    #if param == 'qwdrvn':
    #g = open(f'./{energypath}/H_tot_etat_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_eta1); g.close()
    '''
    if param != 'qwhsbg':
        f = open(f'./{Cxtpath}/Sxtloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); np.save(f, S_loc); f.close()
        g = open(f'./{Cxtpath}/Nxtloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); np.save(g, N_loc); g.close()
    if param != 'qwdrvn':
        g = open(f'./{energypath}/Etloc_t_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); np.save(g, E_loc); g.close()
    if param != 'qwhsbg':
        g = open(f'./{energypath}/Eetatloc_tot_etat_{dtstr}_{epstr[epss]}_{conf}to{conf+1}config.npy','wb'); 
        np.save(g, Eeta_loc); g.close()
    '''

print('processing time = ', time.perf_counter() - start)

#Cxtavg = Cxtavg/nconf
#print('shape of Cxtavg: ', Cxtavg.shape)

