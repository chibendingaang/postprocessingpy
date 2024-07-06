#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nisarg
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.set_printoptions(threshold=2561)

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

Lambda, Mu = map(float,sys.argv[3:5])
begin, end = map(int, sys.argv[5:7])

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss)
choice = int(sys.argv[8])

interval = 1
t_smoothness = int(sys.argv[9]) #typically? 100
if dtsymb ==2: 
    fine_res = 1*t_smoothness
if dtsymb ==1: 
     fine_res = 2*t_smoothness

dtstr = f'{dtsymb}emin3'    
epstr = {3: 'eps_min3', 4: 'eps_min4', 6: 'eps_min6', 8: 'eps_min8' }

if Lambda == 1 and Mu == 0: paramstm = 'hsbg'
elif Lambda == 0 and Mu == 1: paramstm = 'drvn'
else: paramstm = 'a2b0'

alpha = (Lambda - Mu)/(Lambda + Mu)
# single digit suffix vs double digit suffix: 01,02,...,09 v/s 11, 12,..., 99
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr) 

epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}


"""
following statements
 retrieve ~stores the output from obtainspinsnew into a file at given path
"""

if choice == 0:
   param = 'xp' + paramstm
   if paramstm!= 'a2b0': 
       path =  f'./{param}/L{L}/2emin3'
   else: 
       path =  f'./{param}/L{L}/alpha_{alphastr}/2emin3'
    
if choice == 1:
    param = 'qw' + paramstm 
    if paramstm!= 'a2b0': path =   f'./{param}/L{L}/2emin3'
    else:
        #no quadruple precision numerics for generalized case
        param = 'qwa2b0'; path =  f'./{param}/2emin3/alpha_{alphastr}'
    # path = f'./L{L}/{epstr[epss]}/{param}' #path1  
    # the largest file data is currently here: 17-04-2023

if choice ==2:
   if paramstm == 'a2b0': 
       param = 'qw' + paramstm
       path =  f'./{param}/2emin3/alpha_{alphastr}'
   else: 
       param = paramstm
       path =  f'./{param}/L{L}/2emin3'

#for all relevant purposes:
#epss = 3; choice = 0/1; fine_res = t_smoothness = 1

start  = time.perf_counter()

"""settling the steps count discrepancy for good:
Logic for 2*steps + 1:
dt = 0.001 (or 0.002, 0.005)
arrays are stored at 100 step_intervals -> (0.1)*dtsymb in time unit
so steps = (len(array)/(3*L) - 1)*0.1*dtsymb

#Cxtavg = np.zeros((2*steps+1,L)) #2*steps+1
""" 


def calc_Cxt(Cxt_, steps, spin):
    #Cxt_/Cnnxt_ is input zero-value array with the shape (steps/fine_res, L)
    spin = spin[0:steps:fine_res]
    print('mconsv, Cxt, mconsv_sliced pre- and post-slice shapes: \n', spin.shape, Cxt_.shape, spin[10:,2:,:].shape, spin[:(steps//fine_res +1 -10),:(L-2),:].shape) 
    for ti,t in enumerate(range(0,steps,fine_res)):
        # ti: {0 -> steps//fine_res + 1}
        print('time: ', t)
        for x in range(L):
            Cxt_[ti,x] = np.sum(np.sum((spin[ti:, x:, :]*np.roll(np.roll(spin,-x,axis=1),-ti,axis=0)[:(steps//fine_res +1 -ti), :(L-x),:]),axis=2))/((L-x)*(steps//fine_res +1 - ti))  	
    return Cxt_
 
"""
def calc_autoCxt(autoCxt_, steps, spin):
    for ti,t in enumerate(range(0,steps,fine_res)):
        print('time: ', t)
        for x in range(L):
            autoCxt_[ti,x] = np.sum(spin[0,:,:]*np.roll(spin,-x,axis=1),axis=2) #,axis=2))  	
            #multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
            #what does that give us/
    return autoCxt_

def calc_CExt(CExt_, steps, elocal):
    for ti,t in enumerate(range(0,steps,fine_res)):
        for x in range(L):
            CExt_[ti,x] = np.sum((elocal*np.roll(np.roll(elocal,-x,axis=1),-ti,axis=0))[:-ti])/(steps//fine_res+1 - ti)
    return CExt_
"""

def obtaincorrxt(file_j, path): 
    """
    collects the spin configurations a,b for a given initial condition; and finds the time-correlator for them
    step count is kept at every 1/10th of unit time; for example, dt = 0.001 r = 1000
    t = step*r
    
    input: number of steps of the dynamics
    returns: 
    Cxt : the single configuration spin/magnetization (time-)correlator
    Cnnxt :single configuraiton staggered magnetization (time-)correlator
    energ_1 : system energy at time
    energ_eta1 : eta-based system energy at time
    """ 
    
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(begin)))
    steps = int((Sp_aj.size/(3*L))) #0.1*dtsymb multiplication omitted
    print('steps = ', steps)
 
    # reshaped to count spin_xt at every fine_resolved time step

    Sp_a = np.reshape(Sp_aj, (steps,L,3)) #[0:steps:fine_res]; we want the original 961 steps, not 26 or fewer
    print('Sp_a.shape: ' , Sp_a.shape)
    stepcount = min(steps, 2001)
    Sp_a = Sp_a[:stepcount] 

    print('steps , step-jump factor = ', stepcount, fine_res)
    print('Sp_a.shape: ' , Sp_a.shape)
    
    r = int(1./dt)
    j = file_j

    Cxt = np.zeros((stepcount//fine_res+1, L))
    #Cnnxt = np.zeros((stepcount//fine_res+1, L))
    CExt = np.zeros((stepcount//fine_res+1, L))
    #CENxt = np.zeros((stepcount//fine_res+1, L))
    print('Cxt.shape: ', Cxt.shape)
    print('CExt.shape: ', CExt.shape)
    
    Spnbr_a = np.roll(Sp_a,-1,axis=1) 		
    #the next list/array of 3-components along axis=1 has to be shifted up; so -1, not -3
    #np.roll also takes care of Periodic Boundary condition 
    
    #if param == 'qwhsbg':
    E_loc = (-1)*np.sum(Sp_a*Spnbr_a,axis=2) #energy at each site
    energ_1 = np.sum(E_loc, axis = 1) 	#attached the (-1) factor
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    
    Sp_ngva = Sp_a*eta
    Spnbr_ngva = Spnbr_a*eta 
    Eeta_loc = (-1)*np.sum(Sp_a*Spnbr_ngva,axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis = 1) 	#attached (-1) factor
    #array[-:ti] acts like step function: includes matrix products uptil that T-ti


    """
    Definition:
    C(x,t) = 1/(L*(T-t))* \Sum_{x',t'} S_n(x+x',t+t') S_n(x',t')
    """
    # dot product through np.sum(): pick the axis (a*b, axis=2)
    # or just use np.dot(): check the kwargs

    mconsv = np.zeros((stepcount, L, 3)) 
    #mdecay = np.zeros((stepcount, L//2, 3))    
    print('mconsv shape: ', mconsv.shape) 
    '''
    The issue with the transformation below is this:
    we are going from a soft-spin constraint, S^2 = 1, to
    mconsv, which by virtue of the alpha^(-x) factor goes to very large values
    '''
    for ti in range(mconsv.shape[0]):
        for x in range(mconsv.shape[1]):
            mconsv[ti,x] = Sp_a[ti,x]/(alpha)**x
            #mdecay[ti,x] = 0.5*(Sp_a[ti,2*x] - Sp_a[ti,(2*x+1)%L]/alpha)/(alpha)**x
    
    def mconsv_mdecay(param):
        if param == 'qwhsbg' or param == 'xphsbg' or param == 'hsbg':
            return Sp_a, Sp_ngva
        if param == 'qwdrvn' or param == 'xpdrvn' or param == 'drvn':
            return Sp_ngva, Sp_a
        if param == 'xpa2b0' or param == 'qwa2b0':
            return mconsv #, mdecay
    
    
    #return Cxt/L, energ_1 
    '''for ti,t in enumerate(range(0,stepcount,fine_res)):
        for x in range(L):
            Cnnxt[ti,x] = np.sum(np.sum((Sp_ngva*np.roll(np.roll(Sp_ngva,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(stepcount//fine_res+1 - ti) 	
            CENxt[ti,x] = np.sum((Eeta_loc*np.roll(np.roll(Eeta_loc,-x,axis=1),-ti,axis=0))[:-ti])/(stepcount//fine_res+1 - ti)
            #Cnnxt = np.sum(Sp_ngva*Sp_ngva[0,0], axis=2)
    '''

    mconsv = mconsv_mdecay(param)  
    Cxt = calc_Cxt(Cxt, stepcount, mconsv) 
    #stepcount, not steps : 06-07-2024
    print('mconsv.shape: ', mconsv.shape)
    print('Cxt =  \n', Cxt)

    print('shapes of the arrays Cxt/Cnnxt, energ_1/energ_eta1 respectively: \n', Cxt.shape) #, energ_1.shape)
    return Cxt[1:]/L # divide_by_L should be taken care of by calc_Cxt function ideally #, Cnnxt[1:]/L #,energ_1, energ_eta1

if L==1024: Cxtpath = f'Cxt_series_storage/L{L}/{epstr[epss]}'
else: Cxtpath = 'Cxt_series_storage/L{}'.format(L)
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)

nconf = len(range(begin,end))

for conf in range(begin, end):
    # calculate Cxt for each configuration 
    print(conf)
    #Cnnxt, CENxt, energ_1, energ_eta1  = obtaincorrxt(conf, path)
    Cxt = obtaincorrxt(conf, path)
    #if param == 'qwdrvn':
    #    Cnnxt, energ_eta1 = obtaincorrxt(conf)
    #Cxt,Cnnxt,energ_1,energ_eta1, S_loc, N_loc, E_loc, Eeta_loc  = obtaincorrxt(stepcount,conf)
    
    #easier condition: if Cxt (is not None): save the qwhsbg files; if Cnnxt(is not None): save the qwdrvn files;
    #f = open(f'./{Cxtpath}/Cxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, Cxt); f.close()
    if param =='xpa2b0' or param == 'qwa2b0':
        f = open(f'./{Cxtpath}/alpha_ne_pm1/Cxt_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy','wb'); 
        np.save(f, Cxt); 
        f.close()
    
    #ideally, should've been named CExt, and CENxt for storage purposes, but we're continuing it since many files have been 
    #already saved with this name
    #h = open(f'./{CExtpath}/CExt_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(h, CExt); h.close()
    #h = open(f'./{CExtpath}/CEnxt_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(h, CENxt); h.close()
    
    #if param == 'qwhsbg':
    #g = open(f'./{energypath}/H_tot_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_1); g.close()
    #if param == 'qwdrvn':
    #g = open(f'./{energypath}/H_tot_etat_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_eta1); g.close()

print('processing time = ', time.perf_counter() - start)

#Cxtavg = Cxtavg/nconf
#print('shape of Cxtavg: ', Cxtavg.shape)

