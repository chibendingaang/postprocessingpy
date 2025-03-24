#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.interpolate import CubicSpline
import pandas as pd 
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

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
#alphastr = '0pt975'
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

Tth_step = np.concatenate((np.arange(0,100,5), np.arange(100,250,10), np.arange(250, 1250, 25), np.arange(1250,1400,10), np.arange(1400,1501,5)))
T_ = Tth_step.shape[0]
    
def opencorr(L):
    Cxtpath1 = f'Cxt_series_storage/lL{L}/alpha_ne_pm1' #Cxt_series_storage #lL{L} #L{L}
    # alpha_ne_pm1 only for the hybrid, generalized model
    
    if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt' #{param}_cnnxt_{epstr[epss]}_641k.txt' #checkcheck.txt
    #filerepo = f'./{Cxtpath1}/cnnxt_qwdrvn/outcnnxt_545K.txt'
    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{jumpval}.txt'
    # fix the absence of fine_res variable later; fine_res = 10 in the example above
     
    #new calc_corrxt runs:
    #filerepo = f'./{Cxtpath1}_responder/outall.txt'
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])[:8060]#[begin:end]
    f.close()
    f_last = filename[-1] # or pick randomly through np.random.randint(0, len(filename))] 
    #find which file has smallest stepcount, if files are of non-uniform size
    #Cnnxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)
    Cxtk = np.load(f'./{Cxtpath1}/{f_last}', allow_pickle=True)
    steps = int(Cxtk.size/(L+1)) #(2*L-1))
    #check for NaNs
    Cxtavg = np.zeros((T_,L+1)) #2*steps+1
    #Cnnxtavg = np.zeros((steps ,L))

     
    
    for k, fk  in enumerate(filename):
        Cxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)[:steps]
        #keeping the same number of steps as Cxtavg;
        #Note: this may not work when: Cnnxtk.shape[0] < Cxtavg.shape[0]
        if not np.isnan(Cxtk).any(): # proceed only if isnan() method gives False for the Cxt array
            #print('no NaN valued data')
            Cxtavg += Cxtk
            if k%500 ==0: 
                print('prints the 500the configuration: ', Cxtk)
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
    print('Cxtavg.shape: ')
    print(steps, x_)
    countfiles = filenam2.shape[0]

    Cnnxtavg = Cnnxt_
    Cnnxtavg = Cnnxtavg / (countfiles)
    
    # getting the data symmetric w.r.t. the site of max correlation
    Cnnnewxt = Cnnxtavg
    #Cnnnewxt = np.concatenate((Cnnxtavg[:, L-1:L//2:-1], Cnnxtavg[:, 0:L//2]), axis=1)
    #-1 stepping is necessary to mention else the slicing just ends at the last element
    
    print('Cnnxtavg shape: ', Cnnxtavg.shape)
    #t_ = np.array(t * np.array([1/800, 1/200, 1/50, 2/50, 3/50,4/50, 5/50, 8/50]), dtype=int)
    #t_ = np.array([ 8, 12, 16, 24, 32,48]) #([ 16, 24, 32, 48, 64, 96])
    # to create a df, the two dictionaries must contain elements of the samelenght
    # for the alpha_ne_1 case, the concatenate counts an index twice, hence z is of length L-1
    #x = np.concatenate((np.arange(0,x_//2), np.arange(x_//2,x_))) - int(0.5 * x_)
    x_arr = np.arange(-L//2, L//2+1) 
    #avoid using x_//2 as -x_//2 will give -65 for 129 site Cxt 
    
    
    #following are mentioned but not used again
    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization', 'ab': 'Hybrid'}
    #filename = {'stagg': filenam2}
    corr = {'stagg': 'Cnnxt', 'ab': 'Cmhyb_xt', 'magg': "Cxt" }
     
    corrxt_ = np.concatenate((Cnnnewxt[:,L//2+1:], Cnnnewxt[:, :L//2+1]), axis=1) #Cnnnewxt[:t, :L//2+1] #
    print(corrxt_.shape)
    
    #dY_ = np.log(np.array([corrxt_[time, L//2] - corrxt_[time+2, L//2] for time in range(10,steps//(2*jumpval))]))
    #dY_ = dY_[~np.isnan(dY_)]
    #dX_ = np.log(4) + np.log(np.ones(dY_.size)) # 4*np.arange(t)[10] / ( 4*np.arange(t)[steps//(2*jumpval)]))
    #loglog_slope = np.average(dY_/dX_)
    #dY_ = np.log(corrxt_[1*jumpval, L//2] / corrxt_[4*jumpval, L//2]) #steps//(dtsymb*jumpval)	
    #dX_ = np.log( np.arange(t)[1*jumpval] / ( np.arange(t)[4*jumpval])) #steps//(dtsymb*jumpval)
    #loglog_slope = np.round(dY_/dX_, 4)
    #print('slope of d(corrxt(x=0))/dt: ', loglog_slope)
    dX_ = dtsymb*0.1*Tth_step[:-20] #np.arange(corrxt_.shape[0], dtype=np.int32)
    #using Tth_step instead of corrxt_.shape[0] since the former is not uniformly distributed array
    dY_ = corrxt_[:-20, L//2]/corrxt_[0,L//2]
    slope, intercept, r_value, p_value, std_err = linregress(dX_, dY_)


    Cxt_x0 = []
    Cxt_max_xeq = []
    
    for ti,t in enumerate(Tth_step):
        z = corrxt_[ti,:]
        print('time, x, y shapes: ', ti, x_arr.shape, z.shape)
        y = z/corrxt_[0,L//2] #this gives AUC > 1;
        
        Y_smoothened_G = gaussian_filter1d(y, sigma=2)
        Cxt_x0.append(np.max(z))     

        x0_loc = np.argmax(Y_smoothened_G[L//2:]) -L//2-1 #if Cx_.shape is L+1, use L//4 instead of L//2 (shape: 2*L-1)
        Cxt_max_xeq.append(x0_loc)
        
        df = pd.DataFrame({'x_arr': x_arr, 'y':y})  
        window_size = 1 
        #if ti == t_[-1]: 
        #    window_size = 2
        df['y_smooth'] = df['y'].rolling(window=window_size).mean()
        
        print(f'C(0,t = {ti})' , y[L//2])
        auc = auc_simpson(x_arr, y)
        print("area under the curve: ", auc)
        # the averaging at later time steps is pretty bad due to fewer delta_t summations
        # so the auc may not be the same for those time-valued y[t]'s
        #axes.plot(x, y, label=rf'{dtsymb*jumpval*ti}, $a.u.c. = $ {auc:.4f}', linewidth=2)
        #ax2.plot(x *y[L//2],  y/y[L//2], label=f'{dtsymb*jumpval*ti}', linewidth=1) #it was 4*ti earlier because t_ array was defined peculiarly
        #ax2.plot(x / ti**(0.5), ti**(0.5) * y, label=f'{dtsymb*jumpval*ti}', linewidth=1) #it was 4*ti earlier because t_ array was defined peculiarly
        #use -loglog_slope instead of 0.5
        if t in [50,100,200,400,600,800]:#,125,150,200,250]:
            #axes.plot(x_arr, y, label=f'{int(dtsymb*0.1*t)}', linewidth=1.5) #y --> Y_smoothened_G
            axes.plot(df['x_arr'], df['y'], label=f'{int(dtsymb*0.1*t)}', linewidth=1.5)
            ax2.plot(x_arr *y[L//2],  y/y[L//2], linewidth=1) # y--> Y_smoothened_G
         #df['y_smooth']/df['y_smooth'][L//2], label=f'{dtsymb*jumpval*ti}', linewidth=1) #it was 4*ti earlier because t_ array was defined peculiarly
        #ax2.plot(x / ti**(-loglog_slope), ti**(-loglog_slope) * df['y_smooth'], label=f'{dtsymb*jumpval*ti}', linewidth=1) #it was 4*ti earlier
    
    print(f"slope of d(corrxt(x=0))/dt: {slope:.4f}")
    
    #NOTE: depending on the concatenation/indexing of the x_ array, it is either 0 or L//2 to find the peack of 
    #      the Cxt array
    #func= lambda x: a*x**(-0.5)
    #f1 = CubicSpline(4*np.arange(t)[1:-3], corrxt_[1:-3,L//2])
    #f1 = CubicSpline(np.log(4*np.arange(t)[1:-3]), np.log(corrxt_[1:-3,L//2]))(np.log(4*np.arange(t)[1:-3]))
    ax3.loglog(dtsymb*0.1*Tth_step[:-20], corrxt_[:-20, L//2]/corrxt_[0,L//2], '.k', linewidth=1.5)
    #print('f1 shape: ', f1.shape)
    #print('fi: \n',f1)
    #p_opt, p_cov = optimization.curve_fit(func, 4*np.arange(corrxt_.shape[0])[1:-3], f1(4*np.arange(corrxt_.shape[0])[1:-3]))
    #print('popt, pcov: ' , p_opt, p_cov)
    #func_ = p_opt[0]*x**(-0.5)
    #print('func_ shape, func_ : ', func_.shape, '\n', func_)
    #ax3.plot(4*np.arange(corrxt_.shape[0])[1:-3], func_, '-.', linewidth=1.5)
    #ax3.plot(4*np.arange(corrxt_.shape[0])[1:-3], f1, '-.', linewidth=1.5)


    #z2max = (t0)**0.5*corrxt_[t0,L//2]/corrxt_[0,L//2]
    axes.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='center left', ncol=2) # center  right
    axes.set_xlim(max(-32, -3*L//8), min(3*L//8, 32)) #(-min(L//2,125), min(L//2,125))
    #ax2.set_xlim(max(-3*L//8,-33), min(33, 3*L//8) ) #x[7 * L//16], x[9 * L//16 ])
    #ax3.set_xlim(1, min(1000//jumpval, steps//jumpval))
    #ax3.set_ylim(0.025, 1)
    ax2.yaxis.set_label_position("right")
    #ax2.yaxis.tick_right()
    if param == 'qwdrvn': 
        axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    axes.set_xlabel(rf'$\mathbf{{x}} $') #, fontdict=fontlabel)
    axes.set_ylabel(rf'$\mathbf{{C(x,t)}}$')# , fontdict=fontlabel)
    #ax2.set_xlabel(rf'$\mathbf{{x/t^{{{-loglog_slope:.3f}}}}}$') # , fontdict=fontlabel2)
    #ax2.set_ylabel(rf'$\mathbf{{t^-{{{loglog_slope:.3f}}} C(x,t)}}$') #, fontdict=fontlabel2)
    #ax2.set_xlabel(rf'$\mathbf{{xC(0,t)}}$') # , fontdict=fontlabel2)
    #ax2.set_ylabel(rf'$\mathbf{\dfrac{C(x,t)}{C(0,t)}}$') #, fontdict=fontlabel2)
    #ax3.set_xlabel(r'$\mathbf{t}$')#, fontdict=fontlabel2)
    #ax3.set_ylabel(r'$\mathbf{C(0,t)}$') #, fontdict=fontlabel2)
    #axes.set_title(rf'$\alpha = $ {alpha}')
    #ax3.set_title(rf'slope = {loglog_slope:.3f}')
    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_jump{}_{}_{}configs_v5.pdf'.format(corr[magg], L, param, epstr[epss], jumpval,alphastr, countfiles))
    #plt.show()
    


# neither magg or stagg is correct for the hybrid model
plot_corrxt('ab')
# plot_corrxt('stagg')
# plot_corrxt('magg')
