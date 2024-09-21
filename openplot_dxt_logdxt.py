#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nisarg
"""
# Quick and dirty copy of plot_decorravg.py

import sys
import math
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use('matplotlibrc')

#from decorrfly_nnn import getstepcount

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

#int(L*0.05/dt)//32 #25*L//32
Lambda, Mu = map(float, sys.argv[3:5])
begin, end, epss = map(int,sys.argv[5:8])
J2, J1 = map(float,sys.argv[8:10])
#J1 = bool(J2)^1
print('J1, J2: ', J1, J2)

threshfactor = 100
interval = 1 #interval is same as fine_res
inv_dtfact = int(1/(0.1*dtsymb))
dtstr = f'{dtsymb}emin3'
epsilon = 10**(-1.0*epss) #0.00001

J21_ratio = J2/J1
def J21_lamdeci(J21_ratio): 
    if (int(1000*(J21_ratio%1))<10):
        return '00'+str(int(1000*(J21_ratio%1)))
    elif (int(1000*(J21_ratio%1)) < 100):
        return '0' + str(int(1000*(J21_ratio%1)))
    else: 
        return str(int(1000*(J21_ratio%1)))
J21_strdeci = J21_lamdeci(J21_ratio)

if J1 ==0 and J2==0: J1J2comb='invalidinteraction'
elif J1 == 0: J1J2comb = 'J2only'
elif J2 ==0: J1J2comb = 'J1only'
else: J1J2comb = 'J2byJ1_0pt' + J21_strdeci

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr) 

epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}

def getpathnparam():
    if Lambda == 1 and Mu == 0: 
        param = 'qphsbg_NNN'
        path = f'./{param}/L{L}/{dtstr}/{J1J2comb}'
    elif Lambda == 0 and Mu == 1: 
        param = 'qpdrvn_NNN'
        path = f'./{param}/L{L}/{dtstr}/{J1J2comb}'
    #elif Lambda == 1 and Mu == 1: 
    #    param = 'qwa2b0'
    #    path = f'Dxt_storage/{param}' 
        #{epstr[epss]}_{param}'
    else:
        param = 'xpa2b0'
        path = f'./Dxt_storage/alpha_ne_pm1' #{epstr[epss]}_{param}'
    return param, path

param, path = getpathnparam()



if param == 'xpa2b0': 
    filepath = f'./{param}/L{L}/alpha_{alphastr}/{dtstr}'
else:
    if L==1024: filepath = f'./L{L}/eps_min4/{param}'
    else: filepath = f'./{param}/L{L}/{dtstr}/{J1J2comb}'

filepath = path

def getstepcount():
    Sp_aj = np.loadtxt(f'{filepath}/Dxt_{str(11032)}.dat', dtype=np.float128) #pick relevant filenum #4002 for L=130
    steps = int(Sp_aj.shape[0]/(L))
    print('steps: ', steps)
    return steps

steps = getstepcount()

with open(f'{filepath}/outdxt.txt', 'r') as Dxt_repo:
    Dxt_files = np.array([dxtarr for dxtarr in Dxt_repo.read().splitlines()])[begin:end]
    configs = Dxt_files.shape[0]

print(Dxt_files)
    
#steps = 801i
Dxtavg_ = np.zeros((steps,L), dtype=np.float128) 
print('Dxtavg_.shape: ', Dxtavg_.shape)
print('steps : ', steps)

#for Dxtk in Dxt_files:
        
def getDxtavg(Dxt_files, Dxtavg_):
    for k, fk  in enumerate(Dxt_files):
        Dxtk = np.loadtxt(f'{filepath}/{fk}', dtype=np.float128).reshape(steps,L) #when np.load was used:, allow_pickle=True)
        Dxtavg_ += Dxtk #[:inv_dtfact*steps+1]
    return Dxtavg_/len(Dxt_files)
    # why 2*steps + 1: dtsymb = 2*10**(-3); 
    # it is stored every 100 steps in dynamics --> 0.2; 5 time steps per unit time retrieved

Dxtavg_ = getDxtavg(Dxt_files, Dxtavg_)

"""Source of all misery -- To X-shift or not to?"""

Dxtavg = Dxtavg_ #np.concatenate((Dxtavg_[:, L//2:], Dxtavg_[:,0:L//2]), axis=1) 
Dnewxt = Dxtavg[:,L//2:]
logDxt = np.log(Dnewxt/epsilon**2, dtype=np.float64)
"""converting logDxt to np.float64 due to a typical error in the scipy.optimize library, which goes as:
TypeError: Cannot cast array data from dtype('float128') to dtype('float64') according to the rule 'safe'
Traceback (most recent call last):
  File "openplot_dxt_logdxt.py", line 186, in <module>
    kappa, VB = get_VB(cone_spread)
  File "openplot_dxt_logdxt.py", line 178, in get_VB
    popt, pcov = curve_fit(f_empirical, xdata, ydata) #tinit --> t
  File "/apps/codes/anaconda3/lib/python3.7/site-packages/scipy/optimize/minpack.py", line 751, in curve_fit
    res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
  File "/apps/codes/anaconda3/lib/python3.7/site-packages/scipy/optimize/minpack.py", line 394, in leastsq
    gtol, maxfev, epsfcn, factor, diag)
minpack.error: Result from function call is not a proper array of floats.

"""	

np.set_printoptions(threshold=sys.maxsize)

#D_th = 100*epsilon**2		#not needed 

#####  #####
t_,x_ = Dxtavg.shape
#####  #####
#inv_dtfact was needed to reshape the raw Dxt file: however,
#the actual number of steps is dt*raw_steps in the param file

T = np.arange(0,t_) #why the factor of 0.2?
X = -0.5*x_ +  np.arange(0,x_)
#Dnewxt = obtainspins(steps*int(1./dt))
#vel_inst = np.empty(steps*int(1./dt)) 		#not needed

def f_empirical(xbyt,kappa_Lyap, V):
    return kappa_Lyap*(1 - (t_fact*xbyt/V)**2)

#use the following function to guesstimate V_butterfly better
def check_x_intercept(func):
    x_intercepts = []
    for i, fi in enumerate(func[0:-1]):
        if fi*func[i+1] < 0: 
            x_intercepts.append(i)
            break
    if not(x_intercepts): pass 
    else: return x_intercepts[0]

# this variable is just (10/dtsymb)
t_fact = int(10/dtsymb)
#VB = 0
print('t/2 log(Dxt)/t')

"""
Question: how do you determine tinit now given that kappa is predicted with curve_fit?
Easy, just check the t at which log(Dxt/epsilon^2) actually gives a valid x-intercept
"""
"""
def get_tinit():
    for t in range(1, tinit+20):
        print( t, not(check_x_intercept(0.5*t_fact*logDxt[t-1]/t)))
        #prints whether the x_intercept is an empty list or not

get_tinit()
This gives us tinit = 17; 
it may vary a bit across configurations, I guess?
But it still leads to NaN, infinities in the log(Dxt/epsilonsq) array
so it's just guesstimation at this point

"""

#tinit = 100
tinit = int(sys.argv[10]) #11-25; 20 is most likely the origin point of the light-cone
print(Dxtavg[100:tinit+10:10,3*L//8:5*L//8:2])

if L==2048: tfinal = steps-600
else: tfinal = steps-200
cone_spread = np.arange(tinit,tfinal)
#:400 from steps since the cone now covers entire lattice, left to right; 
# calculate this on the basis of vB, which is approx 2.49 for J2/J1 = 0.625


def get_VB(time_range):
    VB = 0; kappa = 0
    for t in time_range:

        xcross =  check_x_intercept(0.5*t_fact*logDxt[t-1]/t)
        v = t_fact*xcross/t
        #default dtype is np.float64
        xdata = np.array([x for x in range(-2 + xcross, 2 + xcross)])/t
        ydata = np.array([ 0.5*t_fact*logDxt[t-1, x]/t for x in range(-2 + xcross, 2+ xcross)])
        
        # above is just v = dx/dt; dx is obtained by calling check_x_intercept func
        popt, pcov = curve_fit(f_empirical, xdata, ydata) #tinit --> t
        kappa += popt[0]
        VB += popt[1]
        # print(' v_i = ', v)
        # VB += v
        # print('t, logdxtbyepssq : ', t, 2.5*logDxt[t-1,0]/t)
    VB = VB/len(time_range)
    kappa = kappa/len(time_range)
    return kappa, VB
    
kappa, VB = get_VB(cone_spread)

#popt, pcov = curve_fit(fun_empir, xdata, 0.5*t_fact*logDxt[tinit-1, 0]/tinit)
#kappa_Lyap = 0.5*t_fact*logDxt[tinit-1, 0]/tinit
print('t_init, Lyapunov exponent kappa, butterfly velocity VB : ', tinit, kappa, VB)


with open(f'kappa_Lyap_{tinit}.txt', 'w') as f:
    f.write(str(kappa))

def plot_decorravg(X, T, Dxtavg, L, steps, param, J1J2comb, dtstr, configs):
    # First plot
    fig, ax = plt.subplots(figsize=(9, 7))  # plt.figure()
    # fig.colorbar(img, orientation='horizontal')
    
    img = ax.pcolormesh(X, dtsymb*T, Dxtavg[:-1, :-1], cmap='seismic', vmin=0, vmax=1.0)
    ax.set_xlabel(r'$\mathbf{x}$')
    ax.set_ylabel(r'$\mathbf{t}$')
    ax.set_ylim(0, dtsymb*steps*8//10)
    # ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16)  # $D(x,t)$ heat map; $t$ v/s $x$;
    # xticks = ticker.MaxNLocator(7)
    # xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
    # ax.set_xticks(xticks)
    
    # plt.colorbar()
    fig.colorbar(img, ax=ax, shrink=0.8)  # , location='left')
    fig.tight_layout()
    plt.savefig(f'./plots/Dxt_L{L}_{param}_{J1J2comb}_dt_{dtstr}_{configs}configs.png')
    # change J1J2comb with alphastr depending on what the file source is
    # plt.show()
    
    """
    # Second plot -- of sublattices
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7))  # plt.figure()
    # fig.colorbar(img, orientation='horizontal')
    
    # due to the initial perturbation being at x==L//2, for 1600, x=800, which in python is xi=799, for x=1602, xi=800
    if L % 4 == 2:
        img1 = ax1.pcolormesh(X[1::2], T, Dxtavg[:-1, 1::2], cmap='seismic', vmin=0, vmax=1.0)
        img2 = ax2.pcolormesh(X[0::2], T, Dxtavg[:-1, 0::2], cmap='seismic', vmin=0, vmax=1.0)
    elif L % 4 == 0:
        img1 = ax1.pcolormesh(X[1::2], T, Dxtavg[:-1, ::2], cmap='seismic', vmin=0, vmax=1.0)
        img2 = ax2.pcolormesh(X[0::2], T, Dxtavg[:-1, 1::2], cmap='seismic', vmin=0, vmax=1.0)
    
    ax1.set_xlabel(r'$\mathbf{x}$')
    ax1.set_ylabel(r'$\mathbf{t}$')
    ax1.set_ylim(0, steps)
    # ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16)  # $D(x,t)$ heat map; $t$ v/s $x$;
    # xticks = ticker.MaxNLocator(7)
    # xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
    # ax.set_xticks(xticks)
    
    ax2.set_xlabel(r'$\mathbf{x}$')
    ax2.set_ylabel(r'$\mathbf{t}$')
    ax2.set_ylim(0, steps)
    # ax.set_title(r'$ \lambda = $ {}, $\mu = $ {}'.format(Lambda, Mu), fontsize=16)  # $D(x,t)$ heat map; $t$ v/s $x$;
    # xticks = ticker.MaxNLocator(7)
    # xticks = np.array([-960, -640, -320, 0, 320, 640, 960])
    # ax.set_xticks(xticks)
    # plt.colorbar()
    
    #fig2.colorbar(img1, ax=[ax1, ax2], shrink=0.8)  # , location='left')
    #fig2.tight_layout()
    plt.savefig(f'./plots/Dxt_splitlattices_L{L}_{param}_{J1J2comb}_dt_{dtstr}_{configs}configs.png')
    # change J1J2comb with alphastr depending on what the file source is
    # plt.show()
    """
    
def plot_logDxt():
    plt.figure()
    fig, axes = plt.subplots(figsize=(8,7))
    x = np.arange(0, L//2)
    #ax2 = axes.inset_axes([0.125, 0.125, 0.525, 0.325])
    #ax2.tick_params(axis='both', which='both', direction='in', width=1.2)
    """ some issue with inset_axes is arising """
    
    t_array = np.array([100, 150, 200,300, 400, 500]) 
    handle1 = []
    for ti,t  in enumerate(int(2/dtsymb)*t_array): #why (2/dtsymb) and not
        ti, = axes.plot(t_fact*x/t, 0.5*t_fact*logDxt[t-1]/t, label=f'{int(t/t_fact)}', linewidth=1.5)
        #ax2.plot(x, Dnewxt[t], linewidth=1)
        handle1.append(ti)
        
    t = 100 #??? shouldn't it be tinit in the old version? and now it's probably not needed
    xbyt = np.arange(0, np.round(1.5*VB, 2), 0.005)
    func_empir = f_empirical(xbyt,kappa, VB)
    #>>> last changes: 21-09-2024; 1910
    # Create a legend for the first line.
    first_legend = axes.legend(handles=handle1, title=r'$t = $', ncol=2, loc='upper right')
    empir_fit, = axes.plot(t_fact*xbyt, func_empir, '--k', label=rf'${{{kappa:0,.2f}}} (1 - (v/{{{VB:0,.2f}}})^2)$',  linewidth=1.5)

    # Add the legend manually to the Axes.
    axes.add_artist(first_legend)
    # Create another legend for the second line.
    axes.legend(handles=[empir_fit], loc='upper center', frameon=False)
    
    #ax1.get_yaxis().set_visible(False)
    #axes.legend(loc = 'upper center')
    #axes.handles=[line2], loc='lower right')
    
    Axlim2 = 3.5 #2.4 for hsbg case
    #ax2.set_xlabel(r'$\mathit{x} $')
    #ax2.set_ylabel(r'$\mathit{D(x,t)} $')
    axes.set_ylim(-0.5, 0.6)
    #ax2.set_xlim(0,ax2lim2)
    #ax2.set_ylim(-0.1, 1.1)
    axes.set_xlabel(r'$\mathbf{x/t} $')
    axes.set_ylabel(r'$\mathbf{\left[ln(D(x,t)/\varepsilon^2)\right]/(2t)} $')
    axes.set_xlim(0,Axlim2)
    #axes.set_ylim(-0.1, 1.1)
    fig.savefig(f'./plots/newlogDxtcomplete_L{L}_{param}_{J1J2comb}_{len(Dxt_files)}configs_{tinit}tinit.pdf')

 
plot_decorravg(X, T, Dxtavg, L, steps, param, J1J2comb, dtstr, configs)
plot_logDxt()

"""
for t in np.arange(0,steps//2+1, 20):
    l = int(4*t)
    ti = t
    logDxt = np.zeros(l)
    counter = 0
    
    for x in range(20,l,20):
        logDxt = np.log(Dxtavg[int(ti),L//2:L//2+l]/epsilon**2)
        f = open('./logDxt/logDxtbyepssq_L{}_t{}_lambda_{}_mu_{}_eps_1emin3_{}config.npy'.format(L,t,Lambda, Mu, len(filename)), 'wb')
        np.save(f, logDxt)
        f.close()
    #print ('t,       log(Dxt/eps*eps)/l' )
    #print()
    #print (t, logDxt/l)
"""
