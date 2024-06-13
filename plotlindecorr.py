#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:38:50 2022

@author: nisarg
"""
import numpy as np
import sys
import numpy.linalg as lin
import math
import time
import matplotlib.pyplot as plt
import scipy.optimize as optimization

start = time.perf_counter()

print('for dt = 0.001, enter 1;    dt = 0.002, enter 2;    dt = 0.005, enter 5')

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
if dtsymb == 1: dt = 0.001; dtstr = '1emin3'
elif dtsymb == 2: dt = 0.002; dtstr = '2emin3'
#elif dtsymb == 5: dt = 0.005 ; dtstr = '5emin3'

steps = int(L*0.05/dt)//32  #(10*L//32 for dt = 0.005)

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4]) #float(sys.argv[1])
begin = int(sys.argv[5])
end = int(sys.argv[6])
interval = 1 #00

if Mu == 1 and Lambda ==0:
    param = 'xpdrvn'
if Mu == 0 and Lambda ==1:
    param = 'xphsbg'

eps = 8 #int(sys.argv[7])
epsilon = 10**(-1.0*eps)

if eps == 3: epstr = '1emin3'
if eps == 8: epstr = '1emin8'
if eps == 4: epstr = '1emin4'
if eps == 6: epstr = '1emin6'

#threshfactor = int(sys.argv[7])
#epspwr = int(sys.argv[8])

#if epspwr == 2: thresh = 'epsq'; epstr = thresh
#elif epspwr == 1: thresh = 'epss'; epstr = thresh

#if threshfactor == 100:

filerepo = 'splitdec100hsbg_{}.txt'.format(dtstr); # threshfactor, param
path = 'Dxt_storage' #.format(thresh,threshfactor) 


f = open(filerepo)
filename = np.array([x for x in f.read().splitlines()])[begin:end] #([x for x in f.readlines()])
f.close()


def get_decorr():
    lindecorr = np.zeros((len(filename[begin:end]),L,2*steps+1))
    for k,fk in enumerate(filename[begin:end]):
        #for k,fk in enumerate(filenum):
        lindecorr[k] = np.load('./{}/{}'.format(path,fk))[0,:last_site]
        
        """if np.all(np.array([~np.isnan(timeregisterDth)])): # x == np.all(...)
            timeregister[k,:] = timeregisterDth 
            #print(x)
        else:
            #necessary to delete empty rows from timeregister array
            #print(~x)
            print(fk)
            np.delete(timeregister, k,0)"""
    avgdecorr = np.average(lindecorr, axis = 0)
    print(np.shape(lindecorr))
    ln_decorr_eps2 = np.log(avgdecorr/epsilon**2)
    return avgdecorr, ln_decorr_eps2
    """
    centraltime = np.zeros((len(filenum),last_site))
    for j in filenum:
        centraltime[j-1] = timeregister[j-1] - avgtime
    """

if param=='hsbg': last_site = 256;
if param == 'drvn': last_site = 256;


 
def plot_logdecorr():
    avgdecorr, ln_decorr_eps2 = get_decorr()
    timearr = np.arange(len(avgdecorr))
    plt.plot(ln_decorr_eps2/(2*timearr), timearr, '--o')
    plt.title('log(Dxt/epsq)/(2t) vs t')
    plt.savefig('./plots/logDxtbyepsq_lin.png')
   
   
    

plot_logdecorr()


stop = time.perf_counter()
print(stop-start)

