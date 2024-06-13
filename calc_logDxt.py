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

steps = 640 #for now; use 1280 #int(L*0.05/dt)//32 
Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

threshfactor = 100
interval = 1
np.set_printoptions(threshold=2561)
#length of the array

if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'
if Lambda == 1 and Mu == 1: param = 'qwa2b0'

dtstr = f'{dtsymb}emin3'
#or use a dictionary
#steps = 1280*L//(1024*dtsymb); #steps = 2560 if L= 2048

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 
#dictionary name
epstr = dict()
epstr[4] = 'eps_min4'
epstr[6] = 'eps_min6'
epstr[8] = 'eps_min8'

#folder addresses
Dxtpath = 'Dxt_storage/L{}'.format(L)
logDxtpath = 'logDxt/L{}'.format(L)

Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1
def obtainspinsnew(steps,file_j): 
    """
    collects the spin configurations a,b for a given initial condition; and finds the decorrelator for them
    step count is kept at every 1/10th of unit time; for example, dt = 0.001 r = 1000
    t = step*r
    
    input: number of steps of the dynamics
    returns: Dnewxt, the symmetric decorrelator
    """ 
    r = int(1./dt)
    Dxt = np.ones((2*steps+1, L), dtype=np.longdouble)
    '''old, extra memory consumingstep
    #Dnewxt = np.empty((steps+1,L)) 
    '''
    j = file_j
    Sp_aj = np.loadtxt(f'./L{L}/{epstr[epss]}/{param}/spin_a_{str(j)}.dat', dtype=np.longdouble)
    Sp_bj = np.loadtxt(f'./L{L}/{epstr[epss]}/{param}/spin_b_{str(j)}.dat', dtype=np.longdouble)
    Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
    Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/(np.sqrt(np.sum(Sp_a*Sp_a, axis=2)*np.sum(Sp_b*Sp_b,axis=2))) 
    #element-wise multiplication
    	
    #Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1)
    #shifting index to site of initial perturbation may not be needed here 
    return Dxt 


def savedecorr():
    """
    stores the output from obtainspinsnew into a file at given path
    """
    samples = range(begin,end)
    nconf = len(samples)
    Dxtavg = np.zeros((2*steps+1, L)) 
    r = int(1./dt);	#this variable isn't used anymore
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    for conf in samples:
        Dxt = obtainspinsnew(steps,conf)
        Dxtavg += Dxt
    Dxtavg = Dxtavg/nconf
    f = open(f'./{Dxtpath}/Dxt_{L}_{param}_dt_{dtstr}_sample_{begin}to{end}.npy', 'wb')
    np.save(f, Dxtavg)
    f.close()
    return Dxtavg 

"""
filelist = 'splitdec{}{}_{}.txt'.format(threshfactor, param, dtstr)
f = open(filelist)
filename = np.array([x for x in f.read().splitlines()])[begin:end]
f.close()
#print(filename)
"""
Dxtavg = savedecorr()
print('shape of Dxtavg: ', Dxtavg.shape)

#vel_inst = np.empty(steps*int(1./dt))
#needed?

#plt.figure()
t_ = np.arange(1, steps+1)
logDxt = np.zeros(t_.shape) 


for x in range(0,128,32):
    print(x)
    for ti,t in enumerate(t_):
        #l = int(2*ti)
        #logDxt = np.zeros(l)
        logDxt[ti] = np.log(Dxtavg[ti,x]/epsilon**2)/(2*t) 
    
    dxtfile = open(f'./{Dxtpath}/Dxt_{x}_array_L{L}_t_{dtstr}_{param}_{epstr[epss]}_{begin}to{end}config.txt', 'wb') 
    #print('Dxt');print(Dxtavg[:,x]/epsilon**2); 
    np.savetxt(dxtfile, Dxtavg[:,x])
    dxtfile.close()
    print(logDxt.shape)
    #print('logDxt'); print(logDxt)
    #logDxt = np.log(Dxtavg[int(ti),x]/epsilon**2)
    
    f = open('./{}/logDxtbyepssq_{}_L{}_t_{}_{}_{}_{}to{}config.npy'.format(logDxtpath,x,L,dtstr,param, epstr[epss],begin,end) ,'wb')
    np.save(f, logDxt)
    f.close()
    #y1 = logDxt/ti #not x1/(2*ti)?? 
    #plt.plot(ti, y1, '-.', linewidth=2, label = '{}'.format(ti))
    #print ('t,       log(Dxt/eps*eps)/l' )
    #print (t, logDxt/l)


#plt.legend()
#plt.grid()
#plt.xlabel(r'$t$')
#plt.ylabel(r'$ln(D(x,t)/\varepsilon^2)$')
#plt.xlim(0,2.8)
#plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
#plt.savefig('./plots/logDxtvs_t_L{}_lambda_{}_mu_{}_{}_{}config.png'.format(L,Lambda, Mu,epstr[epss],nconf))
#plt.show()

#ends here






"""
plt.figure(figsize= (9, 7.5))
#fig.colorbar(img, orientation = 'horizontal')

plt.pcolormesh(Dxtavg[:], cmap = 'seismic',vmin =  0, vmax = 1.0);
plt.xlabel('x')
plt.ylabel(r'$t$')
plt.title(r'$\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu)) #$D(x,t)$ heat map; $t$ v/s $x$;
plt.colorbar()
plt.savefig('./plots/Dxt_L{}_Lambda_{}_Mu_{}_dt_{}_{}confg.png'.format(L,Lambda,Mu,dtstr, len(filename)))
#plt.show()
"""
