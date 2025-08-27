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
t_smoothness = int(sys.argv[9]) #typically? 10
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
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr) 

path_to_directree = f'/home/nisarg/entropy_production/mpi_dynamik/xpa2b0/L{L}/alpha_{alphastr}/{dtstr}'

epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}


"""
following statements
 retrieve ~stores the output from obtainspinsnew into a file at given path
"""

if choice == 0:
   param = 'xp' + paramstm
   if paramstm!= 'a2b0': 
       path =  f'./{param}/L{L}/2emin3'
       path = path_to_directree
   else: 
       path =  f'./{param}/L{L}/alpha_{alphastr}/2emin3'
       path = path_to_directree
   
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
print('param: ', param)
#for all relevant purposes:
#epss = 3; choice = 0/1; fine_res = t_smoothness = 1

start  = time.perf_counter()


def calc_mconsv_ser(spin, alpha):
    alpha_ser = alpha**(-np.arange(L))
    
    #Create 2d matrix explicitly for broadcasting
    alpha_ser = alpha_ser[:, np.newaxis]
    #alpha_2d = np.vstack([alpha_ser] * L)  # Shape: (L, 3)
    print('alpha_2d shape: ', alpha_ser.shape)
    #print(alpha_2d)
    
    # Expand `result_2d` to match the first dimension of `Sp_a`
    #alpha_3d = np.tile(alpha_2d, (Sp_a.shape[0], 1, 1))  # Shape: (arr3.shape[0], 3, 3)
    
    # Perform element-wise multiplication along axes 1 and 2
    mconsv = spin*alpha_ser
    #mconsv[:] = Sp_a[:]*alpha_pow
    print('mconsv shape: ', mconsv.shape) 
            #mdecay[ti,x] = 0.5*(Sp_a[ti,2*x] - Sp_a[ti,(2*x+1)%L]/alpha)/(alpha)**x
    return mconsv


"""settling the steps count discrepancy for good:
Logic for 2*steps + 1:
dt = 0.001 (or 0.002, 0.005)
arrays are stored at 100 step_intervals -> (0.1)*dtsymb in time unit
so 
# Cxt = np.zeros((steps, L))
 where steps = (len(array)/(3*L) - 1)*0.1*dtsymb 
 is the accurate description rather than

#Cxtavg = np.zeros((2*steps+1,L)) #2*steps+1
""" 


def calc_Cxt(Cxt_, steps, spin, alpha):
    #Cxt_/Cnnxt_ is input zero-value array with the shape (steps/fine_res, L)
    # here, spin is the mconsv qty for the alpha-dynamics
    spin = spin[0:steps:fine_res]
    if alpha < 1: alpha_ = 1/alpha
    else: alpha_ = alpha 
    #print('mconsv, Cxt, mconsv_sliced pre- and post-slice shapes: \n', spin.shape, Cxt_.shape, spin[10:,2:,:].shape, spin[:(steps//fine_res +1 -10),:(L-2),:].shape) 
    """
    for ti,t in enumerate(range(0,steps,fine_res)):
        # ti: {0 -> steps//fine_res + 1}
        print('time: ', t)
        for x in range(L):
            Cxt_[ti,x] = np.sum(np.sum((spin*np.roll(np.roll(spin,-x,axis=1),-ti,axis=0))[:(spin.shape[0] -ti), :L-x, :],axis=2))/((L-x)*(spin.shape[0] - ti))  	
    """
    for ti,t in enumerate(range(0,steps,fine_res)):
        # ti: {0 -> steps//fine_res + 1}
        #print('time: ', t)
        T = spin.shape[0]
        spin_t = np.roll(spin, -ti, axis=0)[:T-ti] if ti>0 else spin
        for x in range(-L//2+1,0): #,L): #(-L//2 +1,0) if Cxt_.size was L-1 instead of L+1)
            # consider that this is only open boundary condition here:
            # j = L - x (here, variable x = index i)
            # outer sum over indices i, inner over index j which is shifted/rolled spin
            '''
            spin_x = np.hstack((spin[:,x:], spin[:,:x]))
            spin_tx = np.vstack((spin[ti:], spin[:ti]))[:T-ti, L+x:] #+x  = -np.abs(x)
            Cxt_[ti, x] = np.sum(np.sum(spin[:T-ti, L+x:]*spin_tx, axis=2))/((L-np.abs(x))*(T-ti))
            '''
            Cxt_[ti, x] = np.sum(np.sum((spin_t*np.roll(np.roll(spin_t,-x,axis=1))[:(T -ti), :L-np.abs(x), :],axis=2))/((L-np.abs(x))*(T - ti)) 
            # NOTE: co3fficient of alpha_**(-x) is redundant when mconsv is used as spin
        """     	
        explanation on why midpoint is incorrect: 
        for Open boundary conditions, we just want a half widthed Correlator; 
        so do not copy right half into the left!
        # midpoint = L// 2
        # Copy the first half to the second half
        # Cxt_[:, midpoint:] = Cxt_[:, :midpoint]
        """
        for x in range(0,L//2+1): #(0, L//2)  if Cxt_.size was L-1 instead of L+1)
            #spin_x = np.hstack((spin[:,x:], spin[:,:x]))
            #spin_tx = np.vstack((spin[ti:], spin[:ti]))[:T-ti, :L-x] #+x  = -np.abs(x)
            #Cxt_[ti, x] = np.sum(np.sum(spin[:T-ti, :L-x]*spin_tx, axis=2))/((L-np.abs(x))*(T-ti))
            Cxt_[ti, x] = np.sum(np.sum((spin*np.roll(np.roll(spin,-x,axis=1),-ti,axis=0))[:(T -ti), :L-x, :],axis=2))/((L-x)*(T - ti)) 
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
    stepcount = min(steps, 1281)
    
    # reshaped to count spin_xt at every fine_resolved time step
    Sp_a = np.reshape(Sp_aj, (steps,L,3)) #[0:steps:fine_res]; we want the original 961 steps, not 26 or fewer
    print('Sp_a initial shape: ' , Sp_a.shape)
    Sp_a = Sp_a[:stepcount] 

    print('steps , step-jump factor = ', stepcount, fine_res)
    print('Sp_a shape: ' , Sp_a.shape)
    
    r = int(1./dt)
    j = file_j

    Cxt = np.zeros((stepcount//fine_res+1, L+1))
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

    #mconsv = np.zeros((stepcount, L, 3)) 
    #mdecay = np.zeros((stepcount, L//2, 3))    

    '''
    The issue with the transformation below is this:
    we are going from a soft-spin constraint, S^2 = 1, to
    mconsv, which by virtue of the alpha^(-x) factor goes to very large values
    '''
    # X_ = np.arange(L) #mconsv.shape[1])
    #for i in range(mconsv.shape[1]):
    

    """
    # Create a 3D array with unique elements of shape (100, 16, 3)
    arr3 = np.arange(100 * 16 * 3).reshape(100, 16, 3)
    
    # Create the geometric series of shape (16,)
    geo_series = 0.9 ** np.arange(16)
    
    # Reshape `geo_series` to align with the second dimension (axis=1) of `arr3`
    geo_series = geo_series[:, np.newaxis]  # Shape becomes (16, 1)
    
    # Multiply each (16, 3) subarray along axis=0 with the geometric series
    arr3 = arr3 * geo_series  # Broadcasting happens along the last dimension (axis=2)
    
    print(arr3)
    """
    
    """
    def mconsv_mdecay(param):
        # NOTE: we now have a xphsbg in xpa2b0 folder with alpha_1pt00 folder
        #if param == 'qwhsbg' or param == 'xphsbg' or param == 'hsbg':
        #    return Sp_a, Sp_ngva
        #if param == 'qwdrvn' or param == 'xpdrvn' or param == 'drvn':
        #    return Sp_ngva, Sp_a
        if param == 'xpa2b0' or param == 'qwa2b0' or param == 'xphsbg' or param == 'xpdrvn':
            return mconsv #, mdecay
    """
    mconsv = calc_mconsv_ser(Sp_a, alpha)
    mconsv_tot = np.sum(mconsv, axis=1)
    
    #return Cxt/L, energ_1 
    '''for ti,t in enumerate(range(0,stepcount,fine_res)):
        for x in range(L):
            Cnnxt[ti,x] = np.sum(np.sum((Sp_ngva*np.roll(np.roll(Sp_ngva,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(stepcount//fine_res+1 - ti) 	
            CENxt[ti,x] = np.sum((Eeta_loc*np.roll(np.roll(Eeta_loc,-x,axis=1),-ti,axis=0))[:-ti])/(stepcount//fine_res+1 - ti)
            #Cnnxt = np.sum(Sp_ngva*Sp_ngva[0,0], axis=2)
    '''
    
    #Cxt = calc_Cxt(Cxt, stepcount, Sp_a, alpha)
    Cxt = calc_Cxt(Cxt, stepcount, mconsv, alpha)
    """this was the previous, incorrectly running code:
    # what needs to be modified:
    # 1. the definition of mconsv has to be within the Cxt subroutine where only a single exponent alpha^(-2i-L) takes care of 
    # the alpha factoring
    mconsv = mconsv_mdecay(param)  
    Cxt = calc_Cxt(Cxt, stepcount, mconsv) 
    #stepcount, not steps : 06-07-2024
    print('mconsv.shape: ', mconsv.shape)
    print('Cxt =  \n', Cxt)
    """

    print('shapes of the arrays Cxt/Cnnxt, energ_1/energ_eta1 respectively: \n', Cxt.shape) #, energ_1.shape)
    return Cxt[1:]/L, mconsv_tot # divide_by_L should be taken care of by calc_Cxt function ideally #, Cnnxt[1:]/L #,energ_1, energ_eta1

if L==1024: Cxtpath = f'Cxt_series_storage/L{L}/{epstr[epss]}'
else: Cxtpath = 'Cxt_series_storage/L{}'.format(L)
# pick a constant folder, mentioned below
Cxtpath = f'Cxt_series_storage/lL{L}'
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)

nconf = len(range(begin,end))
Mtotpath = f'H_total_storage/L{L}'

for conf in range(begin, end):
    # calculate Cxt for each configuration 
    print(conf)
    #Cnnxt, CENxt, energ_1, energ_eta1  = obtaincorrxt(conf, path)
    Cxt, mconsv_tot = obtaincorrxt(conf, path_to_directree)
    #if param == 'qwdrvn':
    #    Cnnxt, energ_eta1 = obtaincorrxt(conf)
    #Cxt,Cnnxt,energ_1,energ_eta1, S_loc, N_loc, E_loc, Eeta_loc  = obtaincorrxt(stepcount,conf)
    
    #easier condition: if Cxt (is not None): save the qwhsbg files; if Cnnxt(is not None): save the qwdrvn files;
    #f = open(f'./{Cxtpath}/Cxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, Cxt); f.close()
    if param =='xpa2b0' or param == 'qwa2b0' or param == 'xphsbg':
        f = open(f'./{Cxtpath}/alpha_ne_pm1/Cxt_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy','wb'); 
        np.save(f, Cxt); 
        f.close()
        
        g = open(f'./{Mtotpath}/alpha_ne_pm1/Mconsv_tot_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy', 'wb')
        np.save(g, mconsv_tot)
        g.close()
    
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

