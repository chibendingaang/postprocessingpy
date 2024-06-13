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

#print('for dt = 0.001, enter 1;    dt = 0.002, enter 2;    dt = 0.005, enter 5')

L = int(sys.argv[1])
Lambda = int(sys.argv[2])
Mu = int(sys.argv[3]) #0.99 #float(sys.argv[1])

if Mu == 1 and Lambda ==0:
    param = 'qwdrvn'; last_site = 400; first_site = 100
if Mu == 0 and Lambda ==1:
    param = 'qwhsbg'; last_site = 400; first_site = 100
if Mu ==1 and Lambda ==1:
    param = 'qwa2b0'; last_site = 400; first_site = 100

eps = 3 #int(sys.argv[7])
epsilon = 10**(-1.0*eps)
if eps == 3: epstr = '1emin3'
#if eps == 4: epstr = '1emin4'


beg1 = int(sys.argv[4])
end1 = int(sys.argv[5])
beg2 = int(sys.argv[6])
end2 = int(sys.argv[7])
threshfactor = int(sys.argv[8])
epspwr = int(sys.argv[9])
nbins = int(sys.argv[10])

if epspwr == 2: thresh = 'epsq'; epstr = thresh
elif epspwr == 1: thresh = 'epss'; epstr = thresh

"""
def array_slicer():
	for j, jval in enumerate(timeregister[:,]): 
		if timeregister[j-1,jval] >= 2.5*(j-1): timeregister[j-1,j]
"""


def get_arrivaltime(dtsymb,begin,end):
    if dtsymb == 1: dtstr = '1emin3';
    if dtsymb == 2: dtstr = '2emin3'
    if dtsymb == 3: dtstr = '5emin3' 
    dt = 0.001*dtsymb; steps = int(L*0.05/dt)//32  #(10*L//32 for dt = 0.005)
    #since data is collected at every 100 time steps, t_fact = 1/(timestep*100)
    t_fact = int(10/dtsymb); #t_fact = 5 for dt = 0.002; 10 for dt = 0.001
    
    filerepo = 'split{}{}_{}{}.txt'.format(threshfactor,param,dtstr,epstr); #thresh
    path = dtstr + '/{}/times{}'.format(thresh,threshfactor)  #{}.format(dtstr)
    f = open(filerepo)
    filename = np.array([x for x in f.read().splitlines()])[begin:end] #([x for x in f.readlines()])
    f.close()

    timeregister = np.zeros((len(filename[begin:end]), last_site))
    for k,fk in enumerate(filename[begin:end]):
        #for k,fk in enumerate(filenum):
        timeregisterDth = np.load('./timeregister/{}/{}'.format(path,fk)) #, allow_pickle=True)
        timeregisterDth = timeregisterDth[0,:last_site]
        #since the output timeregister{}{} is stored as a 2-D array, the slicing index are to be kept for that of a 2D_array
        # x = 1 (True) 
        
        if np.all(np.array([~np.isnan(timeregisterDth)])): # x == np.all(...)
            timeregister[k,:] = timeregisterDth 
            #print(x)
        else:
            #necessary to delete empty rows from timeregister array
            #print(~x)
            print(fk)
            np.delete(timeregister, k,0)
    #multiplying timeregister, avgtime with t_fact once here should be enough
    return t_fact, timeregister

    """
    centraltime = np.zeros((len(filenum),last_site))
    for j in filenum:
        centraltime[j-1] = timeregister[j-1] - avgtime
    """

def func2(x,a):    
    return a*x**(1/3.) 

def func(x,a,b):    
    return a*x**(b) 

def gaussf(x,a,b):
    return np.exp(-0.5*np.power(x-a,2)/b**2)# /(b*np.sqrt(2*np.pi))
 
def plot_arrivaltime():
    
    if param=='qwhsbg': prefac = 0.8; vbutf = 1.6
    if param=='qwdrvn': prefac = 0.5; vbutf = 1.3
    xlist = np.linspace(first_site,last_site,7, dtype=int) #100 for qwhsbg #np.array([100,200,300,400,500,600,700]) \
    #np.array([100,200,300,400,500,600,700]) 
    sites = np.arange(0,last_site)
    t_fact1, timeregister_1 = get_arrivaltime(1,beg1,end1)
    t_fact2, timeregister_2 = get_arrivaltime(2,beg2,end2)
    #avgtime_1 = np.average(timeregister_1, axis = 0)/t_fact1; timeregister_1 = timeregister_1/t_fact1
    #avgtime_2 = np.average(timeregister_2, axis = 0)/t_fact1; timeregister_2 = timeregister_2/t_fact2

    print('shape of individual timeregisters: ' , np.shape(timeregister_1), np.shape(timeregister_2))
    timeregister = np.concatenate((timeregister_1/t_fact1, timeregister_2/t_fact2))
    avgtime = np.average(timeregister, axis = 0)
    print('shape of timeregister array after cleaning NaNs: ', np.shape(timeregister))
    #timeregister = timeregister[~np.isnan(timeregister)]
    
    centraltime = np.zeros(np.shape(timeregister)) #(len(filename[begin:end]), len(xlist)))
    print(np.shape(centraltime)) 
    for xi,xp in enumerate(xlist[0:-1]):
        centraltime[:,xi] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(np.shape(timeregister)[0])
        #plt.hist(timeregister[:,xp])
        
        y, binEdges = np.histogram(centraltime[:,xi], density = True, bins = nbins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        """ #the FWHM plot that didn't come to be
        print(y); print(binEdges); print(bincenters)
        #xs = [x for x in range(len(y)) if y[x] > max(y)/2.0]
        #fwhm = max(xs) - min(xs)
        #print ('FWHM for x = {} is {} '.format(xp,fwhm))
        #low, high = np.floor(centraltime[:,xp-1].min()), np.ceil(centraltime[:,xp-1].max())
        #bins = np.linspace(low, high, high - low + 1)
        """
        plt.plot(bincenters, y, '-', label=str(xp))
        plt.xlim((-22,22))
        plt.legend()
        plt.title(r'Histograms for arrival time at different sites')
        
    plt.savefig('./plots/avg_arrivaltime_histogram_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.\
    format(L,Lambda,Mu,epstr, threshfactor, np.shape(timeregister)[0]))
    #plt.show()
    
    centraltavg = np.average(timeregister**2, axis = 0) - avgtime**2
    """print(optimization.curve_fit(func, xdata, ydata, x0, sigma))"""
    print(np.shape(centraltavg))
    plt.figure(figsize = (9,9))
    startsite = first_site
    centraltavg_ = centraltavg[startsite:last_site] #last_site]
    xsites = np.arange(startsite+1,last_site+1 ) #720; last_site+1)
    
    xdata = xsites #- xsites[0] ;
    ydata =  np.sqrt(centraltavg_) # 
    print('optimization parameters a,b and the covariance matrix:')
    p_opt, p_cov = optimization.curve_fit(func, xdata, ydata) #, maxfev=5000)
    par_opt, par_cov = optimization.curve_fit(func2, xdata, ydata)
    print('p_opt, p_cov :', p_opt, p_cov)
    print('par_opt, p_cov :', par_opt, par_cov)
    print('stdev at x = {}:  {}'.format(xsites[-100], centraltavg_[-100]))
    plt.plot(xsites, np.sqrt(centraltavg_), 'o', label='sigma_t_D'); 
    #plt.plot(xsites, np.sqrt(centraltavg_)[0] + np.power(xsites - xsites[0], 1/2.) , \
    # 'b--', label = 'x^{1/2}')
    
    plt.plot(xsites, par_opt[0]*np.power(xsites , 1/3.) , \
    'k--', label = str(par_opt[0])+'x^'+str(1/3)) #+'+'+str(par_opt[1])) #xsites[0] ideally   
    plt.plot(xsites, p_opt[0]*np.power(xsites , p_opt[1]) , \
    'r--', label = str(p_opt[0])+'x^'+str(p_opt[1])) #+'+'+str(p_opt[2]))    
    plt.legend()
    plt.title(r'$\sigma_{t_D}$ vs x')
    plt.savefig('./plots/avg_arrivaltime_stdev_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.\
    format(L,Lambda,Mu,epstr, threshfactor, np.shape(timeregister)[0]))

    plt.figure(figsize= (10,10))
    for j,fj in enumerate(timeregister[0]): #filename):
    	plt.scatter(sites, timeregister[j-1], marker='o', c = 'gray')
    
    plt.scatter(sites,avgtime, color='black', marker = 'o');
    plt.grid()
    plt.xlim(0,last_site); plt.ylim(0,int(last_site/vbutf))
    plt.xlabel('x')
    plt.ylabel(r'$t$')
    plt.title(r'arrival time for $D(x,t)$ front at site $x$; $\lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu))
    plt.savefig('./plots/avg_arrivaltime_vs_x_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,epstr,threshfactor,np.shape(timeregister)[0]))
    plt.show()
    
    xfit = np.arange(-2,2,0.02); #bincenters/xp**(1/3.) 
    gausffit = prefac*gaussf(xfit, 0,np.sqrt(centraltavg[-150])/(xsites[-150]**(1/3.)))
    
    plt.figure(figsize= (11,13))
    for xi,xp in enumerate(xlist[0:]): #[100,200,300,400,500,600,700]:
        centraltime[:,xi] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(np.shape(timeregister)[0])
        y, binEdges = np.histogram(centraltime[:,xi], density = True, bins = nbins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        #centraltime = np.zeros((np.shape(timeregister)[0], last_site), dtype=np.longdouble)
        #centraltime[:,xp-1] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(np.shape(timeregister)[0])
        #y, binEdges = np.histogram(centraltime[:,xp -1], density = True, bins = nbins)
        #bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        plt.subplot(2,1,1)       
        plt.plot(bincenters, y, '-', label=str(xp))
        plt.legend()
        plt.gca().set_xlim((-22,22))
        plt.gca().set_title(r'Histograms for arrival time at different sites')

        print ('bincenters shape : ', np.shape(bincenters))
        #low, high = np.floor(centraltime[:,xp-1].min()), np.ceil(centraltime[:,xp-1].max())
        #bins = np.linspace(low, high, high - low + 1)
        '''#if xp== 50: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--o',  label=str(xp))
        if xp== 100: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--^',  label=str(xp))
        if xp== 200: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--8',  label=str(xp))
        if xp== 300: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--s',  label=str(xp))
        if xp== 400: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--8',  label=str(xp))
        if xp== 500: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--v',  label=str(xp))
        if xp== 600: plt.plot(bincenters/(xp**(p_opt[1])), xp**(p_opt[1])*y, '--v',  label=str(xp))
        plt.gca().set_xlim(-1,1)
        plt.gca().set_title(r'Distribution collapse for the x^{}.PDF vs (t - $\langle t_D0 \rangle $)/x^{}'.format(round(p_opt[1],2),round(p_opt[1],2)));
        '''
        pfit = 1/3. 
        plt.subplot(2,1,2)
        #if xp== 50:  plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--o',  label=str(xp))
        if xi== 0: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--^',  label=str(xp))
        if xi== 1: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--8',  label=str(xp))
        if xi== 2: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--s',  label=str(xp))
        if xi== 3: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--v',  label=str(xp))
        if xi== 4: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--*',  label=str(xp))
        if xi== 5: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--o',  label=str(xp))
        if xi== 6: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--8',  label=str(xp))
        #if xp== 600: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--8',  label=str(xp))
        plt.plot(xfit, gausffit, '--k', linewidth=3)
        #plt.xlabel(r'$(t_{D} - \langle t_{D} \rangle)/x^{b}$')
        #plt.ylabel(r'$x^{b} PDF$')
        plt.legend(); plt.gca().set_xlim(-2,2)
        
        plt.gca().set_title(r'Distribution collapse for the x^{}.PDF vs (t - $\langle t_D0 \rangle $)/x^{}'.\
        format(round(pfit,2),round(pfit,2)));
        
        plt.savefig('./plots/avg_arrivaltime_inset2_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.\
        format(L,Lambda,Mu,epstr,threshfactor, np.shape(timeregister)[0]))
     

    #plt.show()
    

plot_arrivaltime()



stop = time.perf_counter()
print(stop-start)

