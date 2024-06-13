#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

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

def opencorr(L):
    Cxtpath1 = f'Cxt_series_storage/L{L}' #Cxt_series_storage
    if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    filerepo = f'./{Cxtpath1}/cnnxt_qwdrvn/outcnnxt_545K.txt'
    #new calc_corrxt runs:
    #filerepo = f'./{Cxtpath1}_responder/outall.txt'
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])#[begin:end]
    f.close()
    Cxtpath2 = f'{Cxtpath1}/cnnxt_qwdrvn'
    #Cxtpath2 = f'{Cxtpath1}_responder'
    Cnnxtk = np.load(f'./{Cxtpath2}/{filename[-1]}', allow_pickle=True)
    steps = min(int(Cnnxtk.size / L), 1280)

    Cnnxtavg = np.zeros((steps, L))

    for fk in filename:
        Cnnxtk = np.load(f'./{Cxtpath2}/{fk}', allow_pickle=True)[:steps]
        Cnnxtavg += Cnnxtk
    print(steps)
    print('Cxtk first, second entries: ', Cnnxtk[0], Cnnxtk[1]) 
    return Cnnxtavg, filename

def plot_corrxt(magg):
    plt.figure()
    #fontlabel = {'family': 'serif', 'weight': 'bold', 'size': 20}
    #fontlabel2 = {'family': 'serif', 'weight': 'bold', 'size': 16}
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 9))
    ax2 = axes.inset_axes([0.125, 0.475, 0.25, 0.25])
    ax3 = axes.inset_axes([0.675, 0.475, 0.25, 0.25])

    path = f'./{param}/{dtstr}'  # Assuming this path is more relevant for your purpose

    Cnnxt_, filenam2 = opencorr(L)
    steps, x_ = Cnnxt_.shape
    print(steps, x_)
    countfiles = filenam2.shape[0]

    Cnnxtavg = Cnnxt_
    Cnnxtavg = Cnnxtavg / (countfiles)
    Cnnnewxt = np.concatenate((Cnnxtavg[:, L//2:], Cnnxtavg[:, 0:L//2]), axis=1)

    t0 = Cnnxtavg.shape[0]
    t = t0 - t0 % 16
    t_ = np.array(t * np.array([1/8, 1/5, 1/4, 2/5, 1/2, 4/5, 1]), dtype=int)
    t_ = np.array([ 6, 8, 12, 16, 24, 32]) #([ 16, 24, 32, 48, 64, 96])
    x = np.arange(x_) - int(0.5 * x_)
    X = np.arange(x_)

    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization'}
    filename = {'stagg': filenam2}
    corr = {'stagg': 'Cnnxt'}

    corrxt_ = Cnnnewxt[:t]

    for ti in t_:
        y = corrxt_[ti - 1]
        axes.plot(x, y/Cnnnewxt[1,L//2], label=f'{8*ti}', linewidth=2.5)
        ax2.plot(x / ti**(0.5), ti**(0.5) * y/Cnnnewxt[1,L//2], label=f'{8*ti}', linewidth=2)
    dY_ = np.log(corrxt_[1, L//2] / corrxt_[-1, L//2])
    dX_ = np.log( 4*np.arange(t)[1] / ( 4*np.arange(t)[-1]))
    print('slope of d(corrxt(x=0))/dt: ', dY_ / dX_)
    ax3.loglog(4*np.arange(corrxt_.shape[0])[1:], corrxt_[1:, L//2]/corrxt_[1,L//2], '--', linewidth=2)
    tp = t_[0] - 1
    z2max = (tp)**0.5*corrxt_[tp,L//2]/corrxt_[1,L//2]
    axes.legend(title=r'$t = $')
    axes.set_xlim(-125, 125)
    ax2.set_xlim(x[7 * L//16], x[9 * L//16 + 1])
    ax3.set_xlim(1, 400)
    #ax3.set_ylim(0.04, 1)
    if param == 'qwdrvn': 
        axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    axes.set_xlabel(r'$\mathbf{x} $') #, fontdict=fontlabel)
    axes.set_ylabel(r'$\mathbf{C(x,t) }$')# , fontdict=fontlabel)
    ax2.set_xlabel(r'$\mathbf{x/t^{0.5}}$')# , fontdict=fontlabel2)
    ax2.set_ylabel(r'$\mathbf{t^{0.5} C(x,t)}$') #, fontdict=fontlabel2)
    ax3.set_xlabel(r'$\mathbf{t}$')#, fontdict=fontlabel2)
    ax3.set_ylabel(r'$\mathbf{C(0,t)}$') #, fontdict=fontlabel2)

    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_{}configs_v3.pdf'.format(corr[magg], L, param, epstr[epss], countfiles))
    #plt.show()

plot_corrxt('stagg')
# plot_corrxt('magg')
