#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:25:53 2020

@author: nisarg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../matplotlibrc')

L = int(sys.argv[1])
Lambda = int(sys.argv[2])
Mu = int(sys.argv[3])
nconf = int(sys.argv[4])

if Lambda==1 and Mu==0: param = 'hsbg' #nconf = 1100
if Lambda==0 and Mu==1: param = 'drvn' #nconf = 1450
plt.figure()
fig, axes = plt.subplots(figsize = (8,7))



for ti in [80,120,160,200,240]:
    x1 = np.load('../logDxt/logDxtbyepssq_L{}_t{}_lambda_{}_mu_{}_eps_1emin3_{}config.npy'.format(L,ti,Lambda, Mu, nconf))
    #logDxtbyepssq_L512_t100_lambda_1_mu_0_eps_1emin6_100config.npy
    #x1 = np.load('logDxtbyepseps128_{}_{}config.npy'.format(ti,101-1))
    print (np.shape(x1))
    t = np.arange(len(x1))
    y1 = 0.5*x1/(ti) #not x1/(2*ti)??
    #xiarr = []; xarr = []
    #for xi,x in enumerate(x1):
     #   if x*x1[xi+1] <0:
      #      print (x,x1[xi+1], 0.5*(x+x1[xi+1]))
    print(t/ti, x1)
    #print (2*t/ti, y1)
    #plt.figure(figsize = (9,9))
    axes.plot(4*t/ti,y1, label = '{}'.format(ti), linewidth=2) #plt.semilogx

axes.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='upper right', ncol=2)
#plt.grid()
axes.set_xlabel(r'$\mathbf{x/t}$')
axes.set_ylabel(r'$\mathbf{\frac{1}{2t} ln\left(\frac{D(x,t)}{\varepsilon^2}\right)}$')
axes.set_xlim(0,1.8)
#plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
fig.savefig('../plots/logDxtvs_xbyt_L{}_{}_eps_emin3_{}configs.pdf'.format(L,param,nconf))
#plt.show()
'''
plt.figure(figsize = (8,6))
for ti in [20,25,30,35]:
    x1 = np.load('logDxtbyepseps128_{}_{}config.npy'.format(ti,101-1))
    t = np.arange(len(x1))
    plt.plot(t/ti,x1/(2*ti), label = '{}'.format(ti))
plt.legend()
plt.grid()
plt.xlabel(r'$x/t$')
plt.xlim(1.58,1.68)
plt.title('zoominview')
plt.savefig('logDxtvs_zoomed_128.png')
'''

#####3
'''
ti= 20
x1 = np.load('logDxtbyepseps128_{}_{}config.npy'.format(ti,101-1))
t = np.arange(len(x1))
v = 1.63
y = 1 - (t/v*ti)**2
plt.plot(y, x1/(2*ti))'''
