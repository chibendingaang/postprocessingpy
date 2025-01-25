#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.interpolate import CubicSpline
import pandas as pd 

plt.style.use('matplotlibrc')
np.set_printoptions(threshold=2561)

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 * dtsymb
dtstr = f'{dtsymb}emin3'

Lambda, Mu = map(float, sys.argv[3:5])
epss, choice = map(int, sys.argv[5:7])
jumpval = int(sys.argv[7])
#epss, choice = map(int, sys.argv[7:9])

epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
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
    Cxtpath1 = f'Cxt_series_storage/lL{L}/alpha_ne_pm1' #Cxt_series_storage
    # alpha_ne_pm1 only for the hybrid, generalized model
    
    if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    #filerepo = f'./{Cxtpath1}/cnnxt_qwdrvn/outcnnxt_545K.txt'
    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{jumpval}.txt'
    # fix the absence of fine_res variable later; fine_res = 10 in the example above
     
    #new calc_corrxt runs:
    #filerepo = f'./{Cxtpath1}_responder/outall.txt'
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])#[:1152]#[begin:end]
    f.close()
    f_last = filename[-1] # or pick randomly through np.random.randint(0, len(filename))] 
    #find which file has smallest stepcount, if files are of non-uniform size
    #Cnnxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)
    Cxtk = np.load(f'./{Cxtpath1}/{f_last}', allow_pickle=True)
    steps = int(Cxtk.size/(2*L-1))
    #check for NaNs


    Cxtavg = np.zeros((steps,L+1)) #2*steps+1
    #Cnnxtavg = np.zeros((steps ,L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
     
    
    for k, fk  in enumerate(filename):
        Cxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)[:steps]
        #keeping the same number of steps as Cxtavg;
        #Note: this may not work when: Cnnxtk.shape[0] < Cxtavg.shape[0]
        if not np.isnan(Cxtk).any(): # proceed only if isnan() method gives False for the Cxt array
            #print('no NaN valued data')
            Cxtavg += Cxtk
            if k%100 ==0: 
                print('prints the 100the configuration: ', Cxtk)
        else:
            print('NaN value detected at file-index = ', k)

    return Cxtavg, filename
        
    '''
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
    '''

def plot_corrxt(magg):
    plt.figure()
    #fontlabel = {'family': 'serif', 'weight': 'bold', 'size': 20}

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 9))
    ax2 = axes.inset_axes([0.125, 0.675, 0.25, 0.25])
    ax3 = axes.inset_axes([0.675, 0.675, 0.25, 0.25])

    # path = f'./{param}/{dtstr}'  # Assuming this path is more relevant for your purpose

    Cnnxt_, filenam2 = opencorr(L)
    steps, x_ = Cnnxt_.shape #not accurate representation of steps
    print(steps, x_)
    countfiles = filenam2.shape[0]

    Cnnxtavg = Cnnxt_
    Cnnxtavg = Cnnxtavg / (countfiles)
    
    # getting the data symmetric w.r.t. the site of max correlation
    Cnnnewxt = Cnnxtavg
    #Cnnnewxt = np.concatenate((Cnnxtavg[:, L-1:L//2:-1], Cnnxtavg[:, 0:L//2]), axis=1)
    #-1 stepping is necessary to mention else the slicing just ends at the last element

    t0 = Cnnxtavg.shape[0]
    t = t0 - t0 % 16
    # Can we pick longer range of time-steps?
    t_ = np.array(t * np.array([1/16, 1/10, 1/8, 1/5, 1/4, 2/5]), dtype=int)
    #t_ = np.array([ 8, 12, 16, 24, 32,48]) #([ 16, 24, 32, 48, 64, 96])
    # to create a df, the two dictionaries must contain elements of the samelenght
    # for the alpha_ne_1 case, the concatenate counts an index twice, hence z is of length L-1
    #x = np.concatenate((np.arange(0,x_//2), np.arange(x_//2,x_))) - int(0.5 * x_)
    x = np.arange(0, x_//2+1) #-x_//2, 
    X = np.arange(x_//2)
    
    
    #following are mentioned but not used again
    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization', 'ab': 'Hybrid'}
    #filename = {'stagg': filenam2}
    corr = {'stagg': 'Cnnxt', 'ab': 'Cmhyb_xt', 'magg': "Cxt" }
     
    corrxt_ = Cnnnewxt[:t, :L//2+1] # np.concatenate((Cnnnewxt[:t,L//2:], Cnnnewxt[:t, :L//2+1]))

    for ti in t_:
        z = corrxt_[ti,:]
        print('time, x, y shapes: ', ti, x.shape, z.shape)
        y = z/corrxt_[1,0] #this gives AUC > 1;
        df = pd.DataFrame({'x': x, 'y':y})
        
        window_size = 4 
        #if ti == t_[-1]: 
        #    window_size = 2
        df['y_smooth'] = df['y'].rolling(window=window_size).mean()
        auc = auc_simpson(x, y)
        print("area under the curve: ", auc)
        # the averaging at later time steps is pretty bad due to fewer delta_t summations
        # so the auc may not be the same for those time-valued y[t]'s
        axes.plot(x, y, label=rf'{ti}, $a.u.c. = $ {auc:.4f}', linewidth=2)
        ax2.plot(x / ti**(0.5), ti**(0.5) * y, label=f'{ti}', linewidth=1) #it was 4*ti earlier because t_ array was defined peculiarly
        #axes.plot(df['x'], df['y_smooth'], label=f'{ti}', linewidth=2) 
        ax2.plot(x / ti**(0.5), ti**(0.5) * df['y_smooth'], label=f'{ti}', linewidth=1) #it was 4*ti earlier
    dY_ = np.log(corrxt_[10, 0] / corrxt_[steps//(2*jumpval), 0])
    dX_ = np.log( 4*np.arange(t)[10] / ( 4*np.arange(t)[steps//(2*jumpval)]))
    loglog_slope = dY_/dX_
    print('slope of d(corrxt(x=0))/dt: ', loglog_slope)
    
    #NOTE: depending on the concatenation/indexing of the x_ array, it is either 0 or L//2 to find the peack of 
    #      the Cxt array
    #func= lambda x: a*x**(-0.5)
    #f1 = CubicSpline(4*np.arange(t)[1:-3], corrxt_[1:-3,L//2])
    #f1 = CubicSpline(np.log(4*np.arange(t)[1:-3]), np.log(corrxt_[1:-3,L//2]))(np.log(4*np.arange(t)[1:-3]))
    ax3.loglog(4*np.arange(corrxt_.shape[0])[1:-3], corrxt_[1:-3, 0]/corrxt_[1,0], '.k', linewidth=2)
    #print('f1 shape: ', f1.shape)
    #print('fi: \n',f1)
    #p_opt, p_cov = optimization.curve_fit(func, 4*np.arange(corrxt_.shape[0])[1:-3], f1(4*np.arange(corrxt_.shape[0])[1:-3]))
    #print('popt, pcov: ' , p_opt, p_cov)
    #func_ = p_opt[0]*x**(-0.5)
    #print('func_ shape, func_ : ', func_.shape, '\n', func_)
    #ax3.plot(4*np.arange(corrxt_.shape[0])[1:-3], func_, '-.', linewidth=1.5)
    #ax3.plot(4*np.arange(corrxt_.shape[0])[1:-3], f1, '-.', linewidth=1.5)

    tp = t_[0] 
    z2max = (tp)**0.5*corrxt_[tp,L//2]/corrxt_[1,L//2]
    axes.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='lower center', ncol=2) # center  right
    axes.set_xlim(max(-125, -L//4), min(L//4, 125)) #(-min(L//2,125), min(L//2,125))
    ax2.set_xlim(max(-L//16,-33), min(33, L//16) ) #x[7 * L//16], x[9 * L//16 ])
    ax3.set_xlim(1, min(400//jumpval, steps//jumpval))
    ax3.set_ylim(0.025, 1)
    ax2.yaxis.set_label_position("right")
    #ax2.yaxis.tick_right()
    if param == 'qwdrvn': 
        axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    axes.set_xlabel(r'$\mathbf{x} $') #, fontdict=fontlabel)
    axes.set_ylabel(r'$\mathbf{C(x,t)}$')# , fontdict=fontlabel)
    ax2.set_xlabel(r'$\mathbf{x/t^{0.5}}$')# , fontdict=fontlabel2)
    ax2.set_ylabel(r'$\mathbf{t^{0.5} C(x,t)}$') #, fontdict=fontlabel2)
    ax3.set_xlabel(r'$\mathbf{t}$')#, fontdict=fontlabel2)
    ax3.set_ylabel(r'$\mathbf{C(0,t)}$') #, fontdict=fontlabel2)
    axes.set_title(rf'$\alpha = $ {alpha}')
    ax3.set_title(rf'slope = {loglog_slope:.3f}')
    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_{}configs_v5.pdf'.format(corr[magg], L, param, epstr[epss], countfiles))
    #plt.show()


# neither magg or stagg is correct for the hybrid model
plot_corrxt('ab')
# plot_corrxt('stagg')
# plot_corrxt('magg')
