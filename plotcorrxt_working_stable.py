#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
print(sys.executable)

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.interpolate import CubicSpline
import pandas as pd 
from scipy import stats 
from scipy.ndimage import gaussian_filter1d
plt.style.use('matplotlibrc')
plt.rcParams['text.usetex'] = False
np.set_printoptions(threshold=2561)

filter_on = True 

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 * dtsymb
dtstr = f'{dtsymb}emin3'

Lambda, Mu = map(float, sys.argv[3:5])
epss, choice = map(int, sys.argv[5:7])
jumpval = int(sys.argv[7])

epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(np.abs(alpha)%1)))) if (int(100*(np.abs(alpha)%1)) < 10) else (str(int(100*(np.abs(alpha)%1))))
alpha_deci = alphadeci(alpha)
def alphastr_(alpha):
    if alpha>=0:
        return str(int(np.abs(alpha)/1)) + 'pt' + alpha_deci
    else:
        return 'min_'+ str(int(np.abs(alpha)/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr : ', alphastr)

if choice == 0:
    param = 'xpdrvn' if Lambda == 0 and Mu == 1 else 'xphsbg' if Lambda == 1 and Mu == 0 else 'xpa2b0'
    path = f'./xpdrvn/L{L}/{dtstr}'

elif choice == 1:
    param = 'qwdrvn' if Lambda == 0 and Mu == 1 else 'qwhsbg' if Lambda == 1 and Mu == 0 else 'qwa2b0'
    path = f'./{param}/{dtstr}'

elif choice == 2:
    param = 'drvn' if Lambda == 0 and Mu == 1 else 'hsbg' if Lambda == 1 and Mu == 0 else 'qwa2b0'
    path = f'./{param}/{dtstr}'


def auc_simpson(x,f):
    a = x[0]; b = x[-1]; n = len(x)
    h = (b-a)/(n-1)
    auc = (h/3)*(f[0]  + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])
    return auc

def opencorr(L):
    Cxtpath1 = f'Cxt_series_storage/L{L}/alpha_ne_pm1' #lL{L}
    #if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt'
    #if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt'
    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{jumpval}.txt'
    filerepo2 = f'./{Cxtpath1}/outall_energy_{alphastr}_jump{jumpval}.txt'

    for frepo in [filerepo, filerepo2]:
        f = open(frepo)
        filename = np.array([name for name in f.read().splitlines()])#[:248]
        f.close()
        f_last = filename[-1]
        Cxtk = np.load(f'./{Cxtpath1}/{f_last}', allow_pickle=True)
        steps = int(Cxtk.size/(L+1)) #2*L-1

        Cxtavg = np.zeros_like(Cxtk) #((steps,L+1))

        for k, fk  in enumerate(filename):
            Cxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)[:steps]
            if not np.isnan(Cxtk).any():
                Cxtavg += Cxtk
                if k%500 ==0: 
                    print('prints the 500the configuration: ', Cxtk)
                countfiles = filename.shape[0]
            else:
                print('NaN value detected at file-index = ', k)
            if frepo==filerepo: Cxt_= Cxtavg
            if frepo==filerepo2: Cxt_energy = Cxtavg
    return countfiles, Cxt_, Cxt_energy
        

def peak(x, c, sigmasq):
    return np.exp(-np.power(x - c, 2) / sigmasq)

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = np.max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    #np.where(condition, [x,y]) -> x if condition True; y if False
    #eqv to np.nonzero(arr) -> True for nonzero elements
    x2, x1 = lin_interp(x, y, zero_crossings_i[0], half), lin_interp(x, y, zero_crossings_i[1], half)
    return [x1, x2], np.abs(x2-x1)


def plot_corrxt(magg, Cxtavg, conserved_qty):
    plt.figure()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 9))
    ax2 = axes.inset_axes([0.125, 0.675, 0.25, 0.25])
    ax3 = axes.inset_axes([0.675, 0.675, 0.25, 0.25])
    #ax4 = axes.inset_axes([0.675, 0.200, 0.25, 0.25])
    
    steps, x_ = Cxtavg.shape
    print('Cxtavg.shape: ')
    print(steps, x_)

    t0 = Cxtavg.shape[0]
    t = t0 - t0 % 16
    #jumpval = 25
    t_ = np.array([20,40,60,80,100,120]) #np.arange(10,61,10) #array([8,16, 20, 32, 40, 64, 80])
    #t_ = np.array(t * np.array([1/800, 1/200, 1/50, 2/50, 3/50,4/50, 5/50, 8/50]), dtype=int)
    x = np.arange(-x_//2, x_//2)
    X = np.arange(x_//2)
    
    
    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization', 'ab': 'Hybrid'}
    corr = {'stagg': 'Cnnxt', 'ab': 'Cmhyb_xt', 'magg': "Cxt" }
     
    corrxt_ = np.concatenate((Cxtavg[:t,x_//2:], Cxtavg[:t, :x_//2]), axis=1)
    print("corrxt_ shape: (t, # plotting sites)")
    print(corrxt_.shape)
    
    dY_ = np.log(corrxt_[2, L//2] / corrxt_[-8, L//2]) #-0.533, when span: np.arange(t)[2:-3 ]
    dX_ = np.log( np.arange(t)[2] / ( np.arange(t)[-8]))
    loglog_slope = np.round(dY_/dX_, 4)
    print('slope of d(corrxt(x=0))/dt: ', loglog_slope)

    # check fwhm
    fwhm = []
    #for ti in range(0,t):
    #    fwhm.append(half_max_x(x, corrxt_)[1:-3])
    # check C(0,t) inv of the length scale

    # Perform linear regression
    #slope, intercept, r_value, p_value, std_err = stats.linregress(dX_, dY_)

    #print(f"Slope (alpha): {slope:.4f}")
    #print(f"Intercept (log A): {intercept:.4f}")
    #print(f"Standard error of slope: {std_err:.4f}")
    #print(f"R-squared: {r_value**2:.4f}")   
    
    for ti in t_:
        # corrxt_
        z = corrxt_[ti,:]
        print('time, x, y shapes: ', ti, x.shape, z.shape)
        y = z #/corrxt_[0,L//2]
        df = pd.DataFrame({'x': x, 'y':y})
        
        window_size = 2 #{1,2,4,5} #gaussian1D from scipy
        df['y_smooth'] = df['y'].rolling(window=window_size).mean()
        print(f'C(0,t = {ti})' , y[L//2])
        if filter_on:
            Y_smoothened_G = gaussian_filter1d(y, sigma=2)
            auc2 = auc_simpson(x, Y_smoothened_G)
            print("area under the curve for smoothened Y: ", auc2)

            axes.plot(x, Y_smoothened_G, label=f'{int(jumpval/5)*ti}', linewidth=1.5) #dtsymb*jumpval* #ti is the actual timestep
            #axes.plot(df['x'], df['y_smooth'], label=f'{ti}', linewidth=1.5) #dtsymb*jumpval* #ti is the actual timestep
            ax2.plot(x *Y_smoothened_G[L//2],  Y_smoothened_G/Y_smoothened_G[L//2], label=f'{int(jumpval/5)*ti}', linewidth=1)
            #ax2.plot(x / ti**(-loglog_slope), ti**(-loglog_slope) * df['y_smooth'], label=f'{ti}', linewidth=1)
        else:
            axes.plot(x,y, label=f'{int(jumpval/5)*ti}', linewidth=1.5)
            ax2.plot(x*y[L//2], y/y[L//2], label=f'{int(jumpval/5)*ti}', linewidth=1)
        
        auc1 = auc_simpson(x, y)
        print("area under the curve for Y: ", auc1)
    
    #ax4.plot(dtsymb*np.arange(corrxt_.shape[0])[1:-3], fwhm, linewidth=2)
    #plotting the inverse of C(0,t)
    #ax4.loglog(dtsymb*np.arange(corrxt_.shape[0])[1:-3], corrxt_[1,L//2]/corrxt_[1:-3, L//2], '.k', linewidth=1.2) 
    ax3.loglog(dtsymb*int(jumpval/5)*np.arange(corrxt_.shape[0])[1:-3], corrxt_[1,L//2]/corrxt_[1:-3, L//2], '.k', linewidth=1.2) 

    #ax3.loglog(dtsymb*np.arange(corrxt_.shape[0])[1:-3], corrxt_[1:-3, L//2]/corrxt_[1,L//2], '.k', linewidth=2)
    # do not multiply by jumpval in the X-axis array when jumpval = 5; but systemize it
    
    axes.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='lower left', ncol=2)
    axes.set_xlim(max(-32, -64), min(64, 32))
    #ax2.set_xlim(max(-L//8,-33), min(33, L//8) )
    ax2.yaxis.set_label_position("right")
    if param == 'qwdrvn': 
        axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    axes.set_xlabel(rf'$\mathbf{{x}} $')
    axes.set_ylabel(rf'$\mathbf{{C(x,t)}}$')
    ax2.set_xlabel(rf'$\mathbf{{x*C(0,t)}}$') #/t^{{{-loglog_slope:.3f}}}
    ax2.set_ylabel(rf'$\mathbf{{C(x,t)/C(0,t)}}$') #t^-{{{loglog_slope:.3f}}*C(x,t)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("left")
    ax3.set_xlabel(r'$\mathbf{t}$')
    ax3.set_ylabel(rf'$\mathbf{{1/C(0,t)}}$')
    ax3.yaxis.tick_left()
    ax3.yaxis.set_label_position("right")
    axes.set_title(rf'$\alpha = $ {alpha}')
    ax3.set_title(rf'd log(C(0,t))/d log(t) = {loglog_slope:.3f}')
    ax3.title.set_fontsize(9)
    fig.tight_layout()
    if conserved_qty == 'mag':
        plt.savefig('./plots/{}_vs_x_L{}_{}_{}_jump{}_{}_{}configs_v5.pdf'.format(corr[magg], L, param, epstr[epss], jumpval,alphastr, countfiles))
    if conserved_qty == 'energy':
        plt.savefig('./plots/{}_energy_vs_x_L{}_{}_{}_jump{}_{}_{}configs_v5.pdf'.format(corr[magg], L, param, epstr[epss], jumpval,alphastr, countfiles))

if __name__ == "__main__":
    countfiles, Cxt_, Cxt_energy = opencorr(L)
    #countfiles = filenam2.shape[0]
    Cxtavg = Cxt_ / (countfiles)
    Cxt_energy_avg = Cxt_energy/(countfiles)
    plot_corrxt('ab', Cxt_, 'mag')
    plot_corrxt('ab', Cxt_energy, 'energy')

