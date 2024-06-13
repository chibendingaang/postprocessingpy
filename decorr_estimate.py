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
Mu = int(sys.argv[4]) #0.99 #float(sys.argv[1])
begin = int(sys.argv[5])
end = int(sys.argv[6])
interval = 1 #00

if Mu == 1 and Lambda ==0:
    param = 'xpdrvn'
if Mu == 0 and Lambda ==1:
    param = 'xphsbg'

eps = 3 #int(sys.argv[7])
epsilon = 10**(-1.0*eps)

if eps == 3: epstr = '1emin3'
#if eps == 5: epstr = '1emin5'
#if eps == 6: epstr = '1emin6'

'''
#filenuma = np.arange(begin, 1377)
filenumb = np.arange(begin,1910)
filenumc = np.concatenate((filenumb, np.arange(1912, 1984)))
filenumd = np.concatenate((filenumc, np.arange(1995, 2104)))
filenum = np.concatenate((filenumd, np.arange(2105, end)))
#filenum  = np.concatenate((filenumd, np.arange(3002,3205, end)))
#filenum  = np.concatenate((filenume, np.arange(3397, end)))
'''

#Check plot for steps = 3200, 6400, 12800
"""

"""
threshfactor = int(sys.argv[7])
epspwr = int(sys.argv[8])

if epspwr == 2: thresh = 'epsq'; epstr = thresh
elif epspwr == 1: thresh = 'epss'; epstr = thresh

#if threshfactor == 100:

filerepo = 'splitdec{}{}_{}.txt'.format(threshfactor,param,dtstr); #thresh
path = 'Dxt_storage/'.format(thresh,threshfactor)  #{}.format(dtstr)
"""
if threshfactor == 25:
    filerepo = 'split{}{}.txt'.format(threshfactor,param); path = dtstr + '/{}/times{}'.format(thresh, threshfactor)
if threshfactor == 10:
    filerepo = 'split{}{}.txt'.format(threshfactor,param); path = dtstr + '/{}/times{}'.format(thresh, threshfactor)
"""

f = open(filerepo)
filename = np.array([x for x in f.read().splitlines()])[begin:end] #([x for x in f.readlines()])
f.close()


"""
def array_slicer():
	for j, jval in enumerate(timeregister[:,]):
		if timeregister[j-1,jval] >= 2.5*(j-1): timeregister[j-1,j]
"""
def get_arrivaltime():
    timeregister = np.zeros((len(filename[begin:end]), last_site))
    for k,fk in enumerate(filename[begin:end]):
        #for k,fk in enumerate(filenum):
        timeregisterDth = np.load('./timeregister/{}/{}'.format(path,fk))[0,:last_site]
        #since the output timeregister{}{} is stored as a 2-D array, the slicing index are to be kept for that of a 2D_array
        #timeregisterDth = timeregister_Dth[0,:last_site]  print(np.shape(timeregisterDth))
        # x = 1 (True) 
        #timeregisterDth = np.load('./timeregister/{}/timeregister_{}_{}_{}to{}confg.npy'.format(path, L, param, fk,fk+interval))
        if np.all(np.array([~np.isnan(timeregisterDth)])): # x == np.all(...)
            timeregister[k,:] = timeregisterDth 
            #print(x)
        else:
            #necessary to delete empty rows from timeregister array
            #print(~x)
            print(fk)
            np.delete(timeregister, k,0)
    avgtime = np.average(timeregister, axis = 0)
    return timeregister, avgtime
    """
    centraltime = np.zeros((len(filenum),last_site))
    for j in filenum:
        centraltime[j-1] = timeregister[j-1] - avgtime
    """

if param=='xphsbg': last_site = 980;
if param == 'xpdrvn': last_site = 950;

def func2(x,a,c):    
    return a*x**(1/3.) +c 

def func(x,a,b,c):    
    return a*x**(b) + c

 
def plot_arrivaltime():
    xlist = np.array([100,200,300,400,500,600,700,800]) #[100,
    sites = np.arange(0,last_site)
    timeregister, avgtime = get_arrivaltime()
    #print(get_arrivaltime())
    print('shape of timeregister array after cleaning NaNs: ', np.shape(timeregister))
    #timeregister = timeregister[~np.isnan(timeregister)]
    #print(timeregister)

    for xp in xlist:
        centraltime = np.zeros((len(filename[begin:end]), last_site))
        #centralt_avg = np.zeros(last_site)
        centraltime[:,xp-1] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(len(filename))
        #plt.hist(timeregister[:,xp])
        
        y, binEdges = np.histogram(centraltime[:,xp -1], density = True, bins = 12)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        #print(y); print(binEdges); print(bincenters)
        xs = [x for x in range(len(y)) if y[x] > max(y)/2.0]
        fwhm = max(xs) - min(xs)
        #print ('FWHM for x = {} is {} '.format(xp,fwhm))
        #low, high = np.floor(centraltime[:,xp-1].min()), np.ceil(centraltime[:,xp-1].max())
        #bins = np.linspace(low, high, high - low + 1)
        
        plt.plot(bincenters, y, '-', label=str(xp))
        plt.xlim((-160,160))
        plt.legend()
        plt.title(r'Histograms for arrival time at different sites')
        #
        
    plt.savefig('./plots/avg_arrivaltime_histogram_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr, threshfactor, len(filename)))
    #plt.show()
    
    centraltavg = np.average(timeregister**2, axis = 0) - avgtime**2
    """print(optimization.curve_fit(func, xdata, ydata, x0, sigma))"""
    print(np.shape(centraltavg))
    plt.figure(figsize = (9,9))
    centraltavg_ = centraltavg[549:last_site]
    xsites = np.arange(550,last_site+1)
    #print(xsites);
    #params = np.array([1, np.sqrt(centraltavg_[0])])
    #sigma = np.ones(len(xsites))
    
    xdata = xsites - xsites[0] ; ydata =  np.sqrt(centraltavg_) # - centraltavg_[0])
    print('optimization parameters a,b and the covariance matrix: log(y-y0) = log(a) + blog(x-x0) ')
    p_opt, p_cov = optimization.curve_fit(func, xdata, ydata) #, maxfev=5000)
    par_opt, par_cov = optimization.curve_fit(func2, xdata, ydata)
    print('p_opt, p_cov :', p_opt, p_cov)
    print('par_opt, p_cov :', par_opt, par_cov)
    plt.plot(xsites, np.sqrt(centraltavg_), 'o', label='sigma_t_D'); 
    #print('xsites  - xsites[0]: ', xsites - xsites[0]); print(np.sqrt(centraltavg_)); print(np.power(xsites - xsites[0], 1/3.))
    #plt.plot(xsites, np.sqrt(centraltavg_)[0] + np.power(xsites - xsites[0], 1/2.) , 'b--', label = 'x^{1/2}')
    plt.plot(xsites, par_opt[1] + par_opt[0]*np.power(xsites - xsites[0], 1/3.) , 'k--', label = str(par_opt[0])+'x^'+str(1/3)+'+'+str(par_opt[1])) #xsites[0] ideally   
    plt.plot(xsites, p_opt[2] + p_opt[0]*np.power(xsites - xsites[0], p_opt[1]) , 'r--', label = str(p_opt[0])+'x^'+str(p_opt[1])+'+'+str(p_opt[2]))    
    plt.legend()
    plt.title(r'$\sigma_{t_D}$ vs x')
    plt.savefig('./plots/avg_arrivaltime_stdev_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr, threshfactor, len(filename)))
    plt.figure(figsize = (9,9))
    plt.plot(xsites, centraltavg_, '^', label='sigmasq'); plt.legend() 
    plt.title(r'$\sigma_{t_D}^2$ vs x')
    plt.savefig('./plots/avg_arrivaltime_stdevsq_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr, threshfactor, len(filename)))
    #plt.show()
    plt.figure(figsize = (9,9))    
    plt.loglog(xsites, np.sqrt(centraltavg_), 'o', label='sigma');
    plt.loglog(xsites, par_opt[1] + par_opt[0]*np.power(xsites - xsites[0], 1/3) , 'k--', label = str(par_opt[0])+'x^'+str(1/3)) #xsites[0] ideally   
    plt.title(r'$\sigma_{t_D}$ vs x')
    plt.savefig('./plots/avg_arrivaltime_loglogstdev_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr, threshfactor, len(filename)))
    
    plt.figure(figsize= (10,10))
    for j,fj in enumerate(filename):
    	plt.scatter(sites, timeregister[j-1], marker='o', c = 'gray')
    
    plt.scatter(sites,avgtime, color='black', marker = 'o');
    plt.grid()
    plt.xlabel('x')
    plt.ylabel(r'$t$')
    plt.title(r'arrival time for $D(x,t)$ front at site $x$; $\lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu))
    plt.savefig('./plots/avg_arrivaltime_vs_x_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr,threshfactor,len(filename)))
    #plt.show()
    
    
    plt.figure(figsize= (10,10))
    for xp in [550, 650, 750, 850, 950]: #xlist: #[500,600,700,800]: #xlist[3:]:
        centraltime = np.zeros((len(filename), last_site))
        centraltime[:,xp-1] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(len(filename))
        y, binEdges = np.histogram(centraltime[:,xp -1], density = True, bins = 8)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        print ('bincenters shape : ', np.shape(bincenters))
        #ynorm = y.sum()
        #low, high = np.floor(centraltime[:,xp-1].min()), np.ceil(centraltime[:,xp-1].max())
        #bins = np.linspace(low, high, high - low + 1)
        if xp== 420: plt.plot(bincenters, xp**(p_opt[1])*y, 'v',  label=str(xp))
        if xp== 520: plt.plot(bincenters, xp**(p_opt[1])*y, 'o',  label=str(xp))
        if xp== 620: plt.plot(bincenters, xp**(p_opt[1])*y, '^',  label=str(xp))
        if xp== 720: plt.plot(bincenters, xp**(p_opt[1])*y, '*',  label=str(xp))
        if xp== 820: plt.plot(bincenters, xp**(p_opt[1])*y, 's',  label=str(xp))
        if xp== 920: plt.plot(bincenters, xp**(p_opt[1])*y, '8',  label=str(xp))
        plt.xlabel(r'$(t_{D} - \langle t_{D} \rangle)/x^{1/3}$')
        plt.ylabel(r'$x^{1/3} PDF$')
        #plt.xlim((-200,200))
        plt.legend()
        plt.title(r'distribution collapse for arrival times')
        plt.savefig('./plots/avg_arrivaltime_inset2_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr,threshfactor, len(filename)))

    #plt.show()
    

plot_arrivaltime()


stop = time.perf_counter()
print(stop-start)

