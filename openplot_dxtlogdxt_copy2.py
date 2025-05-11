#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use('matplotlibrc')

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 * dtsymb

Lambda, Mu = map(float, sys.argv[3:5])
begin, end, epss = map(int, sys.argv[5:8])
J21_theta = sys.argv[8]
print('J21_theta', J21_theta)

storefreq = 'every200'
if storefreq == 'every200': prefac = 5
elif storefreq == 'every500': prefac = 2
else: prefac = 10

threshfactor = 100
interval = 1
inv_dtfact = int(1/(0.1*dtsymb))
dtstr = f'{dtsymb}emin3'
epsilon = 10**(-1.0 * epss)

J1J2comb = f'J21_theta_{J21_theta}'

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha % 1)))) if (int(100*(alpha % 1)) < 10) else (str(int(100*(alpha % 1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha / 1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr)

epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

def getpathnparam():
    if Lambda == 1 and Mu == 0:
        param = 'qphsbg_NNN'
        path = f'./{param}/L{L}/{dtstr}/{J1J2comb}'
    elif Lambda == 0 and Mu == 1:
        param = 'qpdrvn_NNN'
        path = f'./{param}/L{L}/{dtstr}/{J1J2comb}'
    else:
        param = 'xpa2b0'
        path = f'./Dxt_storage/alpha_ne_pm1'
    return param, path

param, path = getpathnparam()
"""
if param == 'xpa2b0':
    filepath = f'./{param}/L{L}/alpha_{alphastr}/{dtstr}'
else:
    if L == 1024: filepath = f'./L{L}/eps_min4/{param}'
    else: filepath = f'./{param}/L{L}/{dtstr}/{J1J2comb}'
"""
filepath = path

def getstepcount():
    filenum = 33321
    Dxt_j = np.loadtxt(f'{filepath}/Dxt_{filenum}.dat', dtype=np.float128) #Dxt_{storefreq}_
    steps = int(Dxt_j.shape[0] / L)
    print('steps: ', steps)
    return steps

steps = getstepcount()

with open(f'{filepath}/outdxt.txt', 'r') as Dxt_repo: ## outdxt{storefreq[-3:]}.txt
    Dxt_files = np.array([dxtarr for dxtarr in Dxt_repo.read().splitlines()])[begin:end]
    configs = Dxt_files.shape[0]

print(len(Dxt_files))
numconfigs = len(Dxt_files)
print(Dxt_files[0:-1:numconfigs])


Dxtavg_ = np.zeros((steps, L), dtype=np.float128)
print('Dxtavg_.shape: ', Dxtavg_.shape)
print('steps : ', steps)

def getDxtavg(Dxt_files, Dxtavg_):
    for k, fk in enumerate(Dxt_files):
        Dxtk = np.loadtxt(f'{filepath}/{fk}', dtype=np.float128).reshape(steps, L)
        Dxtavg_ += Dxtk
    return Dxtavg_ #/ len(Dxt_files) # let us keep the averaging after calling the function

Dxtavg_ = getDxtavg(Dxt_files, Dxtavg_)

Dxtavg = Dxtavg_/numconfigs
Dnewxt = Dxtavg[:, L//2:]
logDxt = np.log(Dnewxt / epsilon**2, dtype=np.float64)

np.set_printoptions(threshold=sys.maxsize)

t_, x_ = Dxtavg.shape
T = np.arange(0, t_)
X = -0.5 * x_ + np.arange(0, x_)

def f_empirical(xbyt, kappa_Lyap, V):
    return kappa_Lyap * (1 - (t_fact * xbyt / V)**2)

def check_x_intercept(func):
    x_intercepts = []
    for i, fi in enumerate(func[0:-1]):
        if fi * func[i+1] < 0:
            x_intercepts.append(i)
            break
    if not x_intercepts: pass
    else: return x_intercepts[0]

t_fact = 10 #int(prefac / dtsymb) # prefac variable can be made redundant
tinit = int(sys.argv[9])

# here the steps is 2521 for L=1024, dt = 0.002
"""
if L == 2048:
    if dtsymb==2:
        tfinal = min(steps - prefac * 20, 2401) 
    if dtsymb==1:
        tfinal = min(steps - prefac * 20, 4801) 
    
else:
"""
if dtsymb==2:
    tfinal = min(steps - t_fact * 60, 2401) 
if dtsymb==1:
    tfinal = min(steps - t_fact * 120, 4801) 


cone_spread = np.arange(tinit, tfinal, t_fact)

def get_VB(time_range):
    #cone_spread array above is time_range
    VB = 0; kappa = 0; V = 0; p_cov = np.zeros((2, 2))
    counts_None = 0
    
    
    for t in time_range: 
        xcross = check_x_intercept(0.5 * t_fact * logDxt[t - 1] / t)
        
        try:
            if xcross is not None:
                v = t_fact * xcross / t
                V += v
                print(f"Calculated v: {v}")
                
                # The following conditionals are highly empirical:
                if t < t_fact * 25:
                    xdata = np.array([x for x in range(-2 + xcross, 3 + xcross)]) / t
                    ydata = np.array([0.5 * t_fact * logDxt[t - 1, x] / t for x in range(-2 + xcross, 3 + xcross)])
                    popt, pcov = curve_fit(f_empirical, xdata, ydata)
                    kappa += popt[0]
                    VB += popt[1]
                    p_cov += pcov
                elif t < tfinal:
                    xdata = np.array([x for x in range(-3 + xcross, 4 + xcross)]) / t
                    ydata = np.array([0.5 * t_fact * logDxt[t - 1, x] / t for x in range(-3 + xcross, 4 + xcross)])
                    popt, pcov = curve_fit(f_empirical, xdata, ydata)
                    kappa += popt[0]
                    VB += popt[1]
                    p_cov += pcov
            
            else:
                print("xcross is None, skipping calculation.")
                counts_None += 1
        except TypeError as e:
            print(f"TypeError occurred: {e}")
            print("xcross caused a TypeError, skipping calculation.")
            
        #v = t_fact * xcross / t
        

    VB = VB / (len(time_range) - counts_None)
    kappa = kappa / (len(time_range) - counts_None)
    p_cov = p_cov / (len(time_range) - counts_None)
    p_stdev = np.sqrt(np.abs(p_cov))
    return kappa, VB, p_cov, p_stdev

kappa, VB, p_cov, p_stdev = get_VB(cone_spread)
k_err, V_err = p_stdev[0, 0], p_stdev[1, 1]

print('t_init : ', tinit)
print('covariance matrix for kappa, VB: ', p_cov)
print(f'Lyapunov exponent kappa: {kappa:0,.3f}(+/- {k_err:0,.3f})')
print(f'butterfly velocity VB : {VB:0,.3f}(+/- {V_err:0,.3f})')

with open(f'./Kappa_VB/kappa_Lyap_{param}_{J1J2comb}_{end - begin}configs_{tinit}.txt', 'w') as f:
    f.write(f'Lyapunov exponent kappa: {kappa:0,.3f}(+/- {k_err:0,.3f}) \n butterfly velocity VB : {VB:0,.3f}(+/- {V_err:0,.3f}) \n')

def plot_decorravg(X, T, Dxtavg, L, steps, param, J1J2comb, dtstr, configs):
    fig, ax = plt.subplots(figsize=(9, 7))
    img = ax.pcolormesh(X, dtsymb * 2 * T, Dxtavg[:-1, :-1], cmap='seismic', vmin=0, vmax=1.0)
    ax.set_xlabel(r'$\mathbf{x}$')
    ax.set_ylabel(r'$\mathbf{t}$')
    ax.set_ylim(0, dtsymb * 2 * steps)
    fig.colorbar(img, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.savefig(f'./plots/Dxt_L{L}_{param}_{J1J2comb}_dt_{dtstr}_{configs}configs.png')

def plot_logDxt():
    plt.figure()
    fig, axes = plt.subplots(figsize=(8, 7))
    x = np.arange(0, L // 2)
    t_array = np.array([20, 30, 40, 60, 80, 100])
    handle1 = []

    for ti, t in enumerate(t_fact * t_array):
        ti, = axes.plot(t_fact * x / t, 0.5 * t_fact * logDxt[t - 1] / t, label=f'{int(t / t_fact)}', linewidth=1.5)
        handle1.append(ti)
    
    # Again, choice of the Time-step is based on the cone-tip occurence 
    t = 50
    xbyt = np.arange(0, np.round(1.5 * VB, 2), 0.005)
    func_empir = f_empirical(xbyt, kappa, VB)
    first_legend = axes.legend(handles=handle1, title=r'$t = $', ncol=2, loc='upper right')
    empir_fit, = axes.plot(t_fact * xbyt, func_empir, '--k', label=rf'${{{kappa:0,.2f}}} (1 - (v/{{{VB:0,.2f}}})^2)$', linewidth=1.5)
    
    axes.add_artist(first_legend)
    axes.legend(handles=[empir_fit], loc='upper center', frameon=False)
    Axlim2 = np.round(1.8 * VB, 2)
    axes.set_ylim(-0.7, 0.6)
    axes.set_xlabel(r'$\mathbf{x/t} $')
    axes.set_ylabel(r'$\mathbf{\left[ln(D(x,t)/\varepsilon^2)\right]/(2t)} $')
    axes.set_xlim(0, Axlim2)
    fig.savefig(f'./plots/xcross_range/newlogDxtcomplete_L{L}_{param}_{J1J2comb}_{len(Dxt_files)}configs_{tinit}tinit.pdf')

plot_decorravg(X, T, Dxtavg, L, steps, param, J1J2comb, dtstr, configs)
plot_logDxt()
