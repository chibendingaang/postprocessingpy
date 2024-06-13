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
plt.style.use('matplotlibrc')
import pandas as pd

start = time.perf_counter()

#print('for dt = 0.001, enter 1;    dt = 0.002, enter 2;    dt = 0.005, enter 5')
#count the steps correctly; they keep varying based on input config/system size

L = int(sys.argv[1])
#protip: use dtsymb = 1 for qwhsbg and dtsymb = 2 for qwdrvn; data is unclean for dtsymb=1 for the latter case
dtsymb = int(sys.argv[2])
Lambda = int(sys.argv[3])
Mu = int(sys.argv[4]) #0.99 #float(sys.argv[1])


if Mu == 1 and Lambda ==0:
    param = 'qwdrvn'; last_site = 500; first_site = 200 	#another choice could be 100-400
if Mu == 0 and Lambda ==1:
    param = 'qwhsbg'; last_site = 700; first_site = 200
if Mu ==1 and Lambda ==1:
    param = 'qwa2b0'; last_site = 300; first_site = 60



begin = int(sys.argv[5])
end = int(sys.argv[6])

threshfactor = int(sys.argv[7])
eps = int(sys.argv[8])
epsilon = 10**(-1.0*eps)
epstr = f'emin{eps}'

epspwr = int(sys.argv[9])
nbins = int(sys.argv[10])

if epspwr == 2: thresh = 'epsq'; 
elif epspwr == 1: thresh = 'epss'; 

"""
def array_slicer():
	#redundant function to eliminate NaN values from the plotting data
	#NaN values occurred due to the non-detection of front-arrival at a given site
	
	for j, jval in enumerate(timeregister[:,]): 
		if timeregister[j-1,jval] >= 2.5*(j-1): timeregister[j-1,j]
"""
if dtsymb == 1: dtstr = '1emin3';
if dtsymb == 2: dtstr = '2emin3'
if dtsymb == 3: dtstr = '5emin3' 

def get_arrivaltime(begin,end): #(dtsymb,begin,end) if multiple dtstr arrays to be concatenated
    dt = 0.001*dtsymb; 
    steps = int(L*0.05/dt)//32  #(10*L//32 for dt = 0.005)
    #for e.g., when data is collected at every 100 time steps, t_fact = 1/(timestep*100)
    #NOTE: we have a more direct way of determining the steps: 
    #just open a sample spin_a_.dat file and count the number of steps saved as multiples of (1/(100*dt))
    t_fact = int(10/dtsymb)
    
    filerepo = '{}{}out.txt'.format(param,threshfactor); #thresh
    path = 'timeregister/{}/{}/times{}'.format(dtstr,thresh,threshfactor)  #{}.format(dtstr)
    f = open(f'./{path}/{filerepo}')
    filename = np.array([x for x in f.read().splitlines()])[begin:end] #([x for x in f.readlines()])
    f.close()

    timeregister = np.zeros((len(filename[begin:end]), last_site))
    for k,fk in enumerate(filename[begin:end]):
        timeregisterDth = np.load('./{}/{}'.format(path,fk)) #, allow_pickle=True)
        timeregisterDth = timeregisterDth[0,:last_site]
        #since the output timeregister{}{} is stored as a 2-D array, the slicing index is kept for that of a 2D_array
        # x = 1 (True) 
        
        if np.all(np.array([~np.isnan(timeregisterDth)])): # x == np.all(...)
            timeregister[k,:] = timeregisterDth 
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
    
    if param=='qwhsbg': prefac = 0.8; v_butf = 1.65
    if param=='qwdrvn': prefac = 0.525; v_butf = 1.35
    #approximate values of the butterfly velocities we know
    #the prefactors are based on some curve_fitting done apriori I suppose?
    xlist = np.linspace(first_site,last_site,7, dtype=int) #100 for qwhsbg #np.array([100,200,300,400,500,600,700]) \
    
    sites = np.arange(0,last_site)
    t_fact, timeregister = get_arrivaltime(begin,end)
    #timeregister.shape: (len(filename[begin:end]), last_site)
    
    timeregister = timeregister/t_fact
	#averaging over the configurations
    avgtime = np.average(timeregister, axis = 0) 

    print('number of configs in timeregister after cleaning NaNs: ', timeregister.shape[0])
    #timeregister = timeregister[~np.isnan(timeregister)] ---> redundant way; 
    #the not NaN arrays are now removed in the get_arrival() function
    centraltime = np.zeros(np.shape(timeregister)) #(len(filename[begin:end]), len(xlist)))
    centraltavg = np.average(timeregister**2, axis = 0) - avgtime**2
    print("centraltavg.shape: ", centraltavg.shape)    
    centraltavg_ = centraltavg[first_site:last_site]      
    
    #new plot_function
    plt.figure()        
    #font = {'family': 'serif', 'size': 16} #'weight', 'sytle', 'color'	
    fig, ax1 = plt.subplots(figsize=(10,8))
    #fig.suptitle(r'Arrival time distribution for the decorrelator front for individual configurations: $\lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16, fontweight='bold')
    
    for j,fj in enumerate(timeregister[0]): #filename):
    	ax1.scatter(sites, timeregister[j-1], marker='o', c = 'gray')
    
    ax1.scatter(sites,avgtime, color='black', marker = 'o');
    #ax1.grid()
    ax1.set_xlim(0,last_site-50); ax1.set_ylim(0,int(last_site/v_butf))
    ax1.set_xlabel(r'$\mathbf{x}$')
    ax1.set_ylabel(r'$\mathbf{t_{D_0}}$')
    #ax1.set_title(r'Arrival times of decorrelator front $\lambda = {}, \mu = {} $ '.format(Lambda, Mu))
   
    #ax2 = plt.subplot(1,2,2)
    ax2 = ax1.inset_axes([0.125, 0.625, 0.28, 0.28]) #takes up 7.84% of the area of the main plot
    ax3 = ax1.inset_axes([0.675,0.25,0.28,0.28])
    
    xsites = np.arange(first_site+1,last_site+1 ) #720; last_site+1)
    xdata = xsites #- xsites[0] ;
    ydata =  np.sqrt(centraltavg_) # 
    print('optimization parameters a,b and the covariance matrix:')
    p_opt, p_cov = optimization.curve_fit(func, xdata, ydata) #, maxfev=5000)
    par_opt, par_cov = optimization.curve_fit(func2, xdata, ydata)
    print('p_opt, p_cov :', p_opt, p_cov)
    print('par_opt, p_cov :', par_opt, par_cov)
    print('stdev at x = {}:  {}'.format(xsites[-100], centraltavg_[-100]))
    #if param == 'qwdrvn': print('stdev at x = {}:  {}'.format(xsites[-50], centraltavg_[-50]))    
    
    handle1 = []
    for xi,xp in enumerate(xlist[0:-1]):
        centraltime[:,xi] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(np.shape(timeregister)[0])
        #plt.hist(timeregister[:,xp])
        
        y, binEdges = np.histogram(centraltime[:,xi], density = True, bins = nbins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        
        ''' #the FWHM plot that didn't come to be
        print(y); print(binEdges); print(bincenters)
        #xs = [x for x in range(len(y)) if y[x] > max(y)/2.0]
        #fwhm = max(xs) - min(xs)
        #print ('FWHM for x = {} is {} '.format(xp,fwhm))
        #low, high = np.floor(centraltime[:,xp-1].min()), np.ceil(centraltime[:,xp-1].max())
        #bins = np.linspace(low, high, high - low + 1)
        '''
        df = pd.DataFrame({'x': bincenters, 'z':y})
        window_size = 3 #4 
        #if ti == t_[-1]: 
        #    window_size = 2
        df['z_smooth'] = df['z'].rolling(window=window_size).mean()
        #axes.plot(x, z, label=f'{8*ti}', linewidth=2.5)
        xpp, = ax2.plot(df['x'], df['z_smooth'], '-', label=str(xp), linewidth=1.5)
        #xpp, = ax2.plot(bincenters, y, '-', label=str(xp))
        handle1.append(xpp)
        #ax2.set_title(r'Histograms for arrival time at different sites')
        
        centraltime[:,xi] = timeregister[:,xp-1] - avgtime[xp-1]*np.ones(np.shape(timeregister)[0])
        
        pfit = 1/3. 
        xfit = np.arange(-2,2,0.02); #bincenters/xp**(1/3.) 
        gausffit = prefac*gaussf(xfit, 0, np.sqrt(centraltavg[-100])/(xsites[-100]**(1/3.)))
        #if param == 'qwdrvn':
            #gausffit = prefac*gaussf(xfit, 0, np.sqrt(centraltavg[-50])/(xsites[-50]**(1/3.)))

        #if xp== 50:  plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--o',  label=str(xp))
        if xi== 0: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-^',  label=str(xp), linewidth=1)
        if xi== 1: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-8',  label=str(xp), linewidth=1)
        if xi== 2: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-s',  label=str(xp), linewidth=1)
        if xi== 3: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-v',  label=str(xp), linewidth=1)
        if xi== 4: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-*',  label=str(xp), linewidth=1)
        if xi== 5: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-o',  label=str(xp), linewidth=1)
        if xi== 6: ax3.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '-8',  label=str(xp), linewidth=1)
        #if xp== 600: plt.plot(bincenters/(xp**(pfit)), xp**(pfit)*y, '--8',  label=str(xp))
        ax3.plot(xfit, gausffit, '--k', linewidth=2)
        #plt.xlabel(r'$(t_{D} - \langle t_{D} \rangle)/x^{b}$')
        #plt.ylabel(r'$x^{b} PDF$')
        #ax3.set_title(r'Distribution collapse for the x^{}.PDF vs (t - $\langle t_D0 \rangle $)/x^{}'.\
        #format(round(pfit,2),round(pfit,2)));
    ax1.legend(handles=handle1, fontsize=20, frameon=False,title=r'$x = $', ncol=2, loc='lower center')
    ax2.set_xlim((-18,18))
    ax2.set_ylabel(r'p.d.f.')
    ax2.set_xlabel(r'$\mathit{t_{D_0} - \langle t_{D_0} \rangle}$')
    #ax2.legend()
    ax3.set_xlim(-2,2)
    ax3.set_xlabel(r'$\mathit{(t_{D_0} - \langle t_{D_0} \rangle)/x^{1/3}}$')
    ax3.set_ylabel(r"$\mathit{x^{1/3}} . $ p.d.f.")
    plt.savefig('./plots/avg_arrivaltime_vs_x_{}_{}_dt{}_{}_Dth{}_{}bins_{}config.pdf'.format(L,param,dtstr, thresh,threshfactor,nbins,np.shape(timeregister)[0]))
    #plt.savefig('./plots/avg_arrivaltime_histogram_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.\
    #format(L,Lambda,Mu,thresh, threshfactor, np.shape(timeregister)[0]))
    """
    centraltavg = np.average(timeregister**2, axis = 0) - avgtime**2
    #print(optimization.curve_fit(func, xdata, ydata, x0, sigma))
    print(np.shape(centraltavg))
    plt.figure(figsize = (9,9))
    centraltavg_ = centraltavg[first_site:last_site] #last_site]
    xsites = np.arange(first_site+1,last_site+1 ) #720; last_site+1)
    
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
    format(L,Lambda,Mu,thresh, threshfactor, np.shape(timeregister)[0]))

    plt.figure(figsize= (10,10))
    for j,fj in enumerate(timeregister[0]): #filename):
    	plt.scatter(sites, timeregister[j-1], marker='o', c = 'gray')
    
    plt.scatter(sites,avgtime, color='black', marker = 'o');
    plt.grid()
    plt.xlim(0,last_site); plt.ylim(0,int(last_site/v_butf))
    plt.xlabel('x')
    plt.ylabel(r'$t$')
    plt.title(r'arrival time for $D(x,t)$ front at site $x$; $\lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu))
    plt.savefig('./plots/avg_arrivaltime_vs_x_{}_Lambda_{}_Mu{}_{}_Dth_{}_{}config.png'.format(L,Lambda,Mu,thresh,threshfactor,np.shape(timeregister)[0]))
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
        format(L,Lambda,Mu,thresh,threshfactor, np.shape(timeregister)[0]))
     
     """
    #plt.show()
    
plot_arrivaltime()

stop = time.perf_counter()
print(stop-start)

