#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
#a potential source of error could be at line#97 if the file under inspection spin_a_[123].dat is
#absent


import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
np.set_printoptions(threshold=2561)
plt.style.use('matplotlibrc')
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy import integrate
import scipy.optimize as optimize


L = int(sys.argv[1])
L_ = [L]
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb
dtstr = f'{dtsymb}emin3'    

#steps = (1280*L//1024)/dtsymb
#if dt==0.002: dtstr = '2emin3'; steps = 1280
#int(L*0.05/dt)//32 

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])


if Lambda == 1 and Mu == 0: param = 'qwhsbg'
if Lambda == 0 and Mu == 1: param = 'qwdrvn'
if Lambda == 1 and Mu == 1: param = 'qwa2b0'


epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 

epstr = dict()
epstr[3] =  'emin3'
epstr[4] =  'emin4' 
epstr[6] =  'emin6'
epstr[8] =  'emin8'

epstr[33] = 'min3'
epstr[44] = 'min4'
'''choice = int(sys.argv[8])'''

#def peak(x, c, sigmasq):
#    return np.exp(-np.power(x - c, 2) / sigmasq)
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    #np.where(condition, [x,y]) -> x if condition True; y if False
    #eqv to np.nonzero(arr) -> True for nonzero elements
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def auc_trapz(x,f):
    a = x[0]; b = x[-1]; n = len(x)
    h = (b-a)/(n-1)
    auc = 0.5*h*(f[0] + 2*sum(f[1:n-1]) + f[n-1])
    #err =
    return auc

def auc_simpson(x,f):
    a = x[0]; b = x[-1]; n = len(x)
    h = (b-a)/(n-1)
    auc = (h/3)*(f[0]  + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])
    return auc

def fit_func(t,a):
    return a*t**(-0.5) 
# make some trial data


def opencorr(L):
    """
    retrieves ~stores the output from obtainspinsnew into a file at given path
    """
    #Cxtpath = 'Cxt_storage/L{}'.format(L) #Cxt_series_storage
    #energypath = f'H_total_storage/L{L}'
    '''
    if choice ==0:
        path = f'./L{L}/eps_min3/{param}'
    elif choice ==1:
        path = './{}/{}'.format(param,dtstr)        #the epss = 3 for this path

    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(3140))) #str(end)
    steps = int((Sp_aj.size/(3*L))) 
    print('steps = ', steps)
    
    #Cxtavg = np.zeros((16*(steps//640)+1,L)) #2*steps+1
    #Cnnxtavg = np.zeros((16*(steps//640)+1,L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
    '''
    
    path = f'CExt_series_storage/L{L}' #Cxt_storage
    #path = f'Cxt_storage/L{L}'
    
    """I"""
    #filerepo = f'./{path}/{param}_cxt_{dtstr}_{epstr[epss]}_out.txt'
    if param == 'qwhsbg':
        filerepo = f'./{path}/outcxtqwhsbg_545K.txt'
        f = open(filerepo)
        filename = np.array([x for x in f.read().splitlines()]) [begin:end]
        f.close()
    
        fk = filename[-1] 	#np.random.randint(0, len(filename))] 
        #find which file has smallest stepcount, if files are of non-uniform size
        Cxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
        steps = int(Cxtk.size/L)
        print(' #steps = ', steps) # , '\n stepcount = min(steps, 525)' )  
    
        steps = min(steps, 1280)
        print(filename[::100]) 
        Cxtavg = np.zeros((steps, L))
    
        for k, fk  in enumerate(filename):
            Cxtk = np.load(f'./{path}/{fk}', allow_pickle=True)[:steps]
            Cxtavg += Cxtk
        print('Cxtk.shape: ', Cxtk.shape)

    """II"""
    #filerepo2 = f'./{path}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    if param == 'qwdrvn' or param == 'xpdrvn':
        filerepo = f'./{path}/outcnnqwdrvn_545K.txt'
        g = open(filerepo)
        filename = np.array([x for x in g.read().splitlines()])[begin:end]
        g.close()
    
        fk = filename[-1] #np.random.randint(0, len(filename))] 
        Cnnxtk = np.load(f'./{path}/{fk}', allow_pickle=True)
        steps = int(Cnnxtk.size/L)
        print(' #steps = ', steps) # , '\n stepcount = min(steps, 525)' )  
    
        steps = min(steps, 1280)
        print(filename[::100]) 
        Cxtavg = np.zeros((steps, L))
        for k, fk  in enumerate(filename):
            Cnnxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)#[:16*(steps//640)+1]
            Cxtavg += Cnnxtk
        print('Cnnxtk.shape: ', Cnnxtk.shape)  
     
    return filename, Cxtavg #filename2, Cxtk, Cnnxtk #Cxtavg, Cnnxtavg    
    

for L in L_:
    #this will only store the last iteration of the loop
    filename, Cxt_ = opencorr(L)
    steps, x_ = Cxt_.shape
    print(steps,x_)

    Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
    print('Cxtavg.shape = ', Cxtavg.shape)
    Cxtavg = Cxtavg/len(filename)
    
    #print('Cnnxtavg.shape = ', Cnnxtavg.shape)
    #Cnnxtavg = Cnnxtavg/len(filename2)

if param == 'qwhsbg':
    Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
if param == 'qwdrvn' or param == 'xpdrvn':
    Cnnnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)

#the whole point of slicing upto 128 steps
t = Cxtavg.shape[0]
t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array
if param == 'qwdrvn': t_ = np.array([8,12,16,24,32,40])
else: t_ = np.array([16, 24, 32, 48, 64, 96])
#transforming x-axis from 0,L to -L/2, L/2
x = np.arange(x_) - int(0.5*x_)
X = np.arange(x_)
T = np.arange(t)


def plot_corrxt(magg):
    print('magg = ', magg)
    plt.figure()        
    fig, ax1 = plt.subplots(figsize=(11,9))
    magstr = {}
    filename_dic = {}
    corr = {}

    if magg == 'magg':
        corrxt_ = Cnewxt
        magstr['magg'] =  'Energy '
        filename_dic['magg'] = filename
        corr['magg'] = 'CExt'
        
    if magg == 'stagg':
        corrxt_ = Cnnnewxt
        magstr['stagg'] = 'Pseudo-energy'
        filename_dic['stagg'] = filename
        corr['stagg'] = 'CENxt'
    
    
    #ax2 = plt.subplot(1,2,2)
    ax2 = ax1.inset_axes([0.1, 0.66, 0.3, 0.3])
    ax3 = ax1.inset_axes([0.675,0.66,0.3,0.3])
    
    for ti in t_: # [0]
        y = corrxt_[ti-1]
        z = y/corrxt_[1,L//2]
        if param == 'qwhsbg': window_size = 5
        else: window_size = 2
        #if ti == t_[-1]: 
        df = pd.DataFrame({'x': x, 'z':z})
        df['z_smooth'] = df['z'].rolling(window=window_size).mean()
        auc = auc_simpson(X, z)
        print("area under the curve: ", auc)
        ax1.plot(df['x'],df['z_smooth'], label=f'{4*ti}', linewidth=2.5)
        ax2.plot(x/ti**(0.5), ti**(0.5)*df['z_smooth'], label=f'{4*ti}', linewidth=2)
    dY_ = np.log(corrxt_[100, L//2]/corrxt_[20, L//2])
    dX_ = np.log(T[100]/T[20])
    p_opt, p_cov = optimize.curve_fit(fit_func, 4*T[8:-5], corrxt_[8:-5, L//2])
    actual_popt = p_opt[0]/(2*corrxt_[1,L//2])
    func = fit_func(4*T[1:-5], p_opt[0])
    print('slope of dcorrxt/dt: ' , dY_/dX_)
    print('p_opt, p_cov : ', p_opt, p_cov)
    print('first few values of corrxt_ fit: ', func[1:6])
    print('actual_A : ', actual_popt)
    
    ax3.loglog(4*T[1:-5], corrxt_[1:-5,L//2]/corrxt_[1,L//2], '.k', linewidth=1.5)
    curvefit, = ax3.loglog(4*T[1:-5], func/corrxt_[1,L//2], '--', label=rf'${{{actual_popt:0,.3f}}} t^{{{-0.5}}} $', linewidth=1.5)
    ax1.legend(title=r'$\mathit{t} = $', fontsize=20,fancybox=True, shadow=True,frameon=False,  borderpad=1, loc='lower center', ncol=2)
    
    if param == 'qwdrvn': 
        ax1.set_xlim(-36,36) #x[int(14*L/32)], x[int(18*L/32)])
        ax2.set_xlim(-10,10)
    if param == 'qwhsbg': 
        ax1.set_xlim(-160,160) #x[int(14*L/32)], x[int(18*L/32)])
        ax2.set_xlim(-25,25)
    ax2.tick_params(labelsize=16)
    ax3.tick_params(labelsize=16)
    ax1.set_xlabel(r'$\mathbf{x}$'); ax1.set_ylabel(r'$\mathbf{C(x,t)} $') ##C_{NN}(x,t)
    ax2.set_xlabel(r'$\mathit{x/t^{0.5}}$',fontsize=20); ax2.set_ylabel(r'$\mathit{t^{0.5}C(x,t)}$', fontsize=20)
    ax3.set_xlabel(r'$\mathit{t}$',fontsize=20); ax3.set_ylabel(r'$\mathit{C(0,t)}$', fontsize=20)
    ax3.legend(handles=[curvefit], loc='upper center', fontsize=14, frameon=False, fancybox=False, borderpad=0)
   
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_{}_jump{}_{}_{}config_v9.pdf'.format(corr[magg],max(L_),param,1,epstr[epss],filename.shape[0]))
    #plt.show()

if param == 'qwdrvn' or param == 'xpdrvn':plot_corrxt('stagg')
if param == 'qwhsbg': plot_corrxt('magg')

