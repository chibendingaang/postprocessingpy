#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
L = int(sys.argv[1])
#L_ = [L, 512]
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

np.set_printoptions(threshold=2561)

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
choice = int(sys.argv[8])

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


def opencorr(L):
    """
    retrieves ~stores the output from obtainspinsnew into a file at given path
    """
    #Cxtpath = 'Cxt_storage/L{}'.format(L) #Cxt_storage
    #energypath = f'H_total_storage/L{L}'
    if choice ==0:
        path = f'./L{L}/eps_min3/{param}'
    elif choice ==1:
        path = './{}/{}'.format(param,dtstr)        #the epss = 3 for this path

    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(end)))
    steps = int((Sp_aj.size/(3*L))) 
    print('steps = ', steps)
    
    #sneakist piece of the entire code: not consistent as to 
    #what step size to pick
    Cxtavg = np.zeros((min(136,steps),L)) 
    #Cnnxtavg = np.zeros((min(129,steps),L)) #steps//40
    print('Cxtavg.shape: ' , Cxtavg.shape)
 
    path = f'Cxt_series_storage/L{L}' #Cxt_storage
    #path = f'Cxt_storage/L{L}'

    if param == 'qwhsbg':
        filerepo = f'./{path}/{param}_cxt_{dtstr}_{epstr[epss]}_out.txt'
        f = open(filerepo)
        filename = np.array([x for x in f.read().splitlines()]) [begin:end]
        f.close()
        print(filename[::1000]) 
    
        for k, fk  in enumerate(filename):
            Cxtk = np.load(f'./{path}/{fk}', allow_pickle=True)[:min(136,steps)] #steps//40+1
            #print('Cxtk.shape ', Cxtk.shape); 
            Cxtavg += Cxtk
        print('Cxtk.shape: ', Cxtk.shape)
        
    if param == 'qwdrvn':
	    filerepo = f'./{path}/cnnxt_{param}/outcnnxt_545K.txt' #'{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
	    g = open(filerepo)
	    filename = np.array([x for x in g.read().splitlines()]) [begin:end]
	    g.close()
	    print(filename[::1000]) 
	    
	    for k, fk  in enumerate(filename):
	        Cxtk = np.load('./{}/cnnxt_{}/{}'.format(path,param, fk), allow_pickle=True)[:min(steps,136)] #steps//40+1
	        #print('Cxtk.shape ', Cxtk.shape); 
	        Cxtavg += Cxtk
    return filename, Cxtavg # filename1, filename2, Cxtavg, Cnnxtavg    
    #print('steps = ', steps)
    
    """

    print('shape of filename: ', filename1.shape)
   
    
    #for k, fk  in enumerate(filename12):
    #    Sxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[:2*steps+1]
    #    Sxtavg += Sxtk
    #    Sx0t0 += Sxtk[0,0]
    
    #for k, fk  in enumerate(filename22):
    #    Nxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[:2*steps+1]
    #    Nxtavg += Nxtk
    #    Nx0t0 += Nxtk[0,0]
    """    
#Cxtavg = np.zeros((steps//40+1,L)) #2*steps+1

corr = {}

for L in L_:
    filename, Cxt_= opencorr(L)
    steps, x_ = Cxt_.shape
    print(steps,x_)

    if param== 'qwhsbg':
        corr['magg'] = 'Cxt'
        Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
        print('Cxtavg.shape = ', Cxtavg.shape)
        Cxtavg = Cxtavg/len(filename)
        t = Cxtavg.shape[0]
        Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
    #Sxtavg += Sxt_[:,:512,:]/len(filename1); Sx0t0 += Sx0t0_/len(filename1)
    
    if param=='qwdrvn' or param=='xpdrvn':
        corr['stagg'] = 'Cnnxt'
        Cnnxtavg = Cxt_ #[:(16*(steps//16) + 1),:]
        print('Cnnxtavg.shape = ', Cnnxtavg.shape)
        Cnnxtavg = Cnnxtavg/len(filename)
        t = Cnnxtavg.shape[0]
        Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)



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

def func(t,a):
    return a*t**(-0.5) 
# make some trial data

#the whole point of slicing upto 128 steps

#t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4)))
#t_ = np.array((t-t%16)*np.array([ 3/64, 4/64, 3/32, 4/32, 3/16, 4/16]), dtype=int)
t_ = np.array([6, 8, 10, 12, 16, 20, 32])
print('time steps for the plot: ', t_)

#transforming x-axis from 0,L to -L/2, L/2
x = np.arange(x_) - int(0.5*x_)
X = np.arange(x_)

def plot_corrxt(magg):
    if magg == 'magg':
        #filename = filename1
        corrxt_ = Cnewxt
        magstr = 'Magnetization'
    if magg == 'stagg':
        #filename = filename2
        corrxt_ = Cnnnewxt
        magstr = 'Staggered magnetization'
    
    plt.figure()    
    #font = {'family': 'serif',  # choose the appropriate font family 'serif': ['Times'],  # choose the appropriate serif font 'size': 12}
    
    fontlabel = {'family': 'serif', 'size': 16} #'weight', 'style', 'color'	
    fontlabel2 = {'family': 'serif', 'size': 12} #'weight', 'style', 'color'	
    fig, ax1 = plt.subplots(figsize=(11,9))
    #fig.suptitle(f'{magstr} correlations')
    #ax2 = plt.subplot(1,2,2)
    ax2 = ax1.inset_axes([0.1, 0.525, 0.3, 0.3])
    ax3 = ax1.inset_axes([0.675,0.525,0.3,0.3])
    
    p_opt, p_cov = optimization.curve_fit(func, np.arange(t), corrxt_[:,L//2])
    print('curve_fit params: ', p_opt, p_cov)
    #area_uc = []
    
    for ti in t_:    
        y = corrxt_[ti-1]
        auc = auc_simpson(X, y)
        print("area under the curve: ", auc)
        ax1.plot(x,y, label=f'{40*ti}')
        ax2.plot(x/ti**(0.5), ti**(0.5)*y, label=f'{40*ti}')
    ax3.loglog(np.arange(t)[1:], corrxt_[1:,L//2], '-.', linewidth=2)

    ax1.legend(); 
    ax1.set_xlim(x[4*L//16], x[12*L//16+1])
    ax2.set_xlim(x[7*L//16], x[9*L//16+1])
    #ax3.set_xlim(1, 500)
    ax1.set_xlabel(r'$x$', fontdict=fontlabel); ax1.set_ylabel(r'$C(x,t) $', fontdict=fontlabel) ##C_{NN}(x,t)
    ax2.set_xlabel(r'$x/t^{0.5}$', fontdict=fontlabel2); ax2.set_ylabel(r'$t^{0.5}$C(x,t)', fontdict=fontlabel2)
    ax3.set_xlabel(r't', fontdict=fontlabel2); ax3.set_ylabel(r'C(0,t)', fontdict=fontlabel2)
    
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    plt.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_lambda_{}_mu_{}_{}_{}config.pdf'.format(corr[magg],max(L_),Lambda, Mu,epstr[epss],len(filename)))
    #plt.show()

if param == 'qwhsbg': plot_corrxt('magg')
if param == 'qwdrvn' or param == 'xpdrvn':
    plot_corrxt('stagg')

