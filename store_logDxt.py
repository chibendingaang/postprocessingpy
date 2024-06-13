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


if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'

if Lambda == 1 and Mu == 1:
    param = 'qwa2b0'
    if dt==0.002: dtstr = '2emin3'; steps = 1280
    elif dt==0.005: dtstr = '5emin3'; steps = 640
    elif dt==0.001: dtstr = '1emin3'; steps = 2560

if dt== 0.001: dtstr = '1emin3' 
if dt==0.002: dtstr = '2emin3'
if dt==0.005: dtstr = '5emin3'

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 
#directory names 
if epss == 4: 
	epstr = 'eps_min4'; 
if epss == 6: 
	epstr = 'eps_min6';
if epss == 8: 
	epstr = 'eps_min8';


Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1


Dxtpath = 'Dxt_storage/L{}'.format(L)
def savedecorr():
    """
    stores the output from obtainspinsnew into a file at given path
    """
    r = int(1./dt); 
    #D_th = math.pow(10,-2); D_th2 = math.pow(10,-4)
    Dnewxt = obtainspinsnew(steps)
    Dxtavg += Dnewxt
 
    f = open('./{}/Dxt_{}_{}_dt_{}_sample_{}to{}.npy'.format/
             (Dxtpath,L,param, dtstr,base_ + begin,base_ + end), 'wb') #Mu, epstr
    np.save(f, Dnewxt)
    f.close()    

"""
filerepo = 'splitdec{}{}_{}.txt'.format(threshfactor, param, dtstr)
f = open(filerepo)
filename = np.array([x for x in f.read().splitlines()])[begin:end]
f.close()
#print(filename)
"""

for conf in range(begin, end):
    Dxt = obtainspinsnew(steps,conf)
    Dxtavg += Dxt

#Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1
#for k, fk  in enumerate(filename):
#	Dxtk = np.load('./{}/{}'.format(Dxtpath, fk), allow_pickle=True);
#	Dxtavg += Dxtk

nconf = len(range(begin,end))
Dxtavg = Dxtavg/nconf
print('shape of Dxtavg: ', Dxtavg.shape)

#Dnewxt = obtainspins(steps*int(1./dt))
vel_inst = np.empty(steps*int(1./dt))

logDxtpath = 'logDxt/L{}'.format(L)

plt.figure()
t_ = np.arange(0, steps+1)
for x in range(0,96,32):
    #l = int(2*ti)
    #logDxt = np.zeros(l)
    #counter = 0    
    logDxt = np.log(Dxtavg[:,x]/epsilon**2)
    print(logDxt)
    #logDxt = np.log(Dxtavg[int(ti),x]/epsilon**2)
    print(logDxt.shape)
    f = open('./{}/logDxtbyepssq_L{}_t{}_{}_{}_{}config.npy'.format(logDxtpath,L,dtstr,param, epstr,nconf) ,'wb')
    np.save(f, logDxt)
    f.close()
    #y1 = logDxt/ti #not x1/(2*ti)?? 
    #plt.plot(ti, y1, '-.', linewidth=2, label = '{}'.format(ti))
    #print ('t,       log(Dxt/eps*eps)/l' )
    #print (t, logDxt/l)


plt.legend()
plt.grid()
plt.xlabel(r'$t$')
plt.ylabel(r'$ln(D(x,t)/\varepsilon^2)$')
#plt.xlim(0,2.8)
plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
plt.savefig('./plots/logDxtvs_t_L{}_lambda_{}_mu_{}_{}_{}config.png'.format(L,Lambda, Mu,epstr,nconf))
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
