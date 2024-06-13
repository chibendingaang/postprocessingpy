#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import integrate
import scipy.optimize as optimize
import pandas as pd 

plt.style.use('matplotlibrc')
np.set_printoptions(threshold=2561)

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 * dtsymb
dtstr = f'{dtsymb}emin3'

Lambda, Mu, epss, choice = map(int, sys.argv[3:7])
#epss, choice = map(int, sys.argv[7:9])

epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

if choice == 0:
    param = 'xpdrvn' if Lambda == 0 and Mu == 1 else 'xphsbg' if Lambda == 1 and Mu == 0 else 'xpa2b0'
    path = f'./xpdrvn/L{L}/{dtstr}'

elif choice == 1:
    param = 'qwdrvn' if Lambda == 0 and Mu == 1 else 'qwhsbg' if Lambda == 1 and Mu == 0 else 'qwa2b0'
    path = f'./{param}/{dtstr}'

elif choice == 2:
    param = 'drvn' if Lambda == 0 and Mu == 1 else 'hsbg' if Lambda == 1 and Mu == 0 else 'qwa2b0'
    path = f'./{param}/{dtstr}'

def fit_func(x,a):
    return a*x**(-0.5)

def opencorr(L):
    ''' figure out consistent path of storage: too many file names mentioned below here'''
    Cxtpath1 = f'Cxt_series_storage/L{L}' #Cxt_series_storage
    if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    #filerepo = f'./{Cxtpath1}/cnnxt_qwdrvn/outcnnxt_545K.txt'
    #new calc_corrxt runs:
    filerepo = f'./{Cxtpath1}/outall_{param}.txt'
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])#[begin:end]
    f.close()
    Cxtpath2 = f'{Cxtpath1}' #/cnnxt_qwdrvn'
    if param == 'qwdrvn': Cxtpath2 = f'{Cxtpath1}/cnnxt_qwdrvn'
    Cxtk = np.load(f'./{Cxtpath2}/{filename[-1]}', allow_pickle=True)
    if param == 'qwhsbg': steps = min(int(Cxtk.size / L), 129) #129 for most cases
    else: steps = min(int(Cxtk.size / L), 641) #129 for most cases

    Cxtavg = np.zeros((steps, L))

    for fk in filename:
        Cxtk = np.load(f'./{Cxtpath2}/{fk}', allow_pickle=True)[:steps]
        Cxtavg += Cxtk
    print(steps)
    print('Cxtk first, second entries: ', Cxtk[0], Cxtk[1]) 
    return Cxtavg, filename

def plot_corrxt(magg):
    plt.figure()
    #fontlabel = {'family': 'serif', 'weight': 'bold', 'size': 20}
    #fontlabel2 = {'family': 'serif', 'weight': 'bold', 'size': 16}
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 9))
    ax2 = axes.inset_axes([0.1, 0.66, 0.3, 0.3])
    ax3 = axes.inset_axes([0.675, 0.66, 0.3, 0.3])

    path = f'./{param}/{dtstr}'  # Assuming this path is more relevant for your purpose

    Cnnxt_, filenam2 = opencorr(L)
    steps, x_ = Cnnxt_.shape
    print(steps, x_)
    countfiles = filenam2.shape[0]

    Cnnxtavg = Cnnxt_
    Cnnxtavg = Cnnxtavg / (countfiles)
    Cnnnewxt = np.concatenate((Cnnxtavg[:, L//2:], Cnnxtavg[:, 0:L//2]), axis=1)

    t0 = Cnnxtavg.shape[0]
    trange = t0 - t0 % 16
    t_ = np.array(t0 * np.array([1/8, 1/5, 1/4, 2/5, 1/2, 4/5, 1]), dtype=int)
    if param == 'qwdrvn': t_ = np.array([ 8, 12, 16, 24, 32,40]) 
    if param == 'qwhsbg': t_ = np.array([ 16, 24, 32, 48, 64, 96])
    x = np.arange(x_) - int(0.5 * x_)
    X = np.arange(x_)

    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization'}
    filename = {'stagg': filenam2}
    corr = {'stagg': 'Cnnxt', 'magg': 'Cxt'}

    corrxt_ = Cnnnewxt[:trange]
    T = np.arange(trange)

    for ti in t_:
        y = corrxt_[ti ]
        z = y/corrxt_[1,L//2]
        df = pd.DataFrame({'x': x, 'z':z})
        if param == 'qwhsbg': window_size = 2 #4 
        if param == 'qwdrvn': window_size = 5
        #if ti == t_[-1]: 
        #    window_size = 2
        df['z_smooth'] = df['z'].rolling(window=window_size).mean()
        #axes.plot(x, z, label=f'{8*ti}', linewidth=2.5)
        axes.plot(df['x'], df['z_smooth'], label=f'{4*ti}', linewidth=2.5)
        ax2.plot(x / ti**(0.5), ti**(0.5) * df['z_smooth'], label=f'{4*ti}', linewidth=2)
    dY_ = np.log(corrxt_[5, L//2] / corrxt_[-5, L//2])
    dX_ = np.log( 4*T[5] / ( 4*T[-5]))
    print('slope of d(corrxt(x=0))/dt: ', dY_ / dX_)
    
    ax3.loglog(4*T[1:-5], corrxt_[1:-5, L//2]/corrxt_[1,L//2], '.k', linewidth=1.5)
    #form: optimize.curve_fit(function, xdata, ydata , bounds=)
    p_opt, p_cov = optimize.curve_fit(fit_func, 4*T[5:-5], corrxt_[5:-5, L//2])
    func_ = fit_func(4*T[1:-5],p_opt[0]) 
    print('Corrxt[t,L//2] first few array values :', corrxt_[:,L//2][1:6])
    print('func_ shape, func_ : ', func_.shape, '\n func:', func_)
    
    actual_popt = p_opt[0]/(2*corrxt_[1,L//2]) #factor of sqrt(4) corresponding to t_fact
    print('popt, pcov: ', p_opt, p_cov)
    print('actual_A : ', actual_popt)
    curvefit, = ax3.loglog(4*T[1:-5], func_/corrxt_[1,L//2], '--', label=rf'${{{actual_popt:0,.3f}}} t^{{{-0.5}}} $', linewidth=1.5)

    tp = t_[0] 
    z2max = (tp)**0.5*corrxt_[tp,L//2]/corrxt_[1,L//2]
    axes.legend(title=r'$\mathit{t} = $', fontsize=20, fancybox=True,borderpad=1,shadow=True, frameon=False, loc='lower center', ncol=2)
    ax3.legend(handles=[curvefit], loc='upper center', fontsize=14, frameon=False, fancybox=False)
    axes.set_xlim(-160, 160)
    ax2.set_xlim(x[7 * L//16], x[9 * L//16 + 1])
    if param == 'qwdrvn': ax3.set_xlim(4, 320)
    else: ax3.set_xlim(4, 500)
    
    #ax3.set_ylim(0.04, 1)
    if param == 'qwdrvn': 
        axes.set_xlim(-80, 80); ax2.set_xlim(-15,15)
    axes.set_xlabel(r'$\mathbf{x} $') #, fontdict=fontlabel)
    axes.set_ylabel(r'$\mathbf{C(x,t)}$')# , fontdict=fontlabel)
    ax2.tick_params(labelsize=16)
    ax3.tick_params(labelsize=16)
    ax2.set_xlabel(r'$\mathit{x/t^{0.5}}$', fontsize=20)# , fontdict=fontlabel2)
    ax2.set_ylabel(r'$\mathit{t^{0.5} C(x,t)}$',fontsize=20) #, fontdict=fontlabel2)
    ax3.set_xlabel(r'$\mathit{t}$',fontsize=20)#, fontdict=fontlabel2)
    ax3.set_ylabel(r'$\mathit{C(0,t)}$',fontsize=20) #, fontdict=fontlabel2)

    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_{}configs_v9.pdf'.format(corr[magg], L, param, epstr[epss], countfiles))
    #plt.show()

if Lambda == 1 and Mu == 0: plot_corrxt('magg')
elif Lambda == 0 and Mu == 1: plot_corrxt('stagg')
