#!/usr/bin/env python3
# -*- codiddf utf-8 -*-
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
import scipy.fft as sff

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
#choice={0 : 'xpdrvn', 1: 'qwdrvn'}

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

def func(t,a):
    return a*t**(-0.5) 
# make some trial data


def opencorr(L):
    """
    retrieves ~stores the output from obtainspinsnew into a file at given path
    outputs: the filename and avg of cmm/nnxt arrays
    """
    #Cxtpath = 'Cxt_storage/L{}'.format(L) #Cxt_storage
    #energypath = f'H_total_storage/L{L}'
    if choice ==0: #corrsponds to xpdrvn precision 
        path = f'./L{L}/eps_min3/{param}'
    elif choice ==1: #corresponds to qwdrvn
        path = f'./{param}/{dtstr}' # be consistent .format(param,dtstr)        
        #the epss = 3 for this path

    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(end+800)))
    steps = int((Sp_aj.size/(3*L))) 
    print('steps = ', steps)
    
    Cxtavg = np.zeros(((steps//40)+1,L)) #[:131] #2*steps+1
    Cnnxtavg = np.zeros(((steps//40)+1,L))# [:131]
    print('Cxtavg.shape: ' , Cxtavg.shape)
 
    path = f'Cxt_series_storage/L{L}' #Cxt_storage
    #path = f'Cxt_storage/L{L}'

    #if param == 'qwhsbg':
    filerepo = f'./{path}/{param}_cxt_{dtstr}_{epstr[epss]}_out.txt'
    f = open(filerepo)
    filename1 = np.array([x for x in f.read().splitlines()]) [begin:end]
    f.close()
    print(filename1[::100]) 
    
    '''for k, fk  in enumerate(filename1):
        Cxtk = np.load(f'./{path}/{fk}', allow_pickle=True)[:(steps//40)+1]
        #print('Cxtk.shape: ', Cxtk.shape)
        Cxtavg += Cxtk
    '''
    
    filerepo = f'./{path}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    g = open(filerepo)
    filename2 = np.array([x for x in g.read().splitlines()]) [begin:end]
    g.close()
    print(filename2[::100]) 
    
    for k, fk  in enumerate(filename2):
        Cnnxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[:(steps//40)+1]
        Cnnxtavg += Cnnxtk
    return filename1, filename2, Cxtavg, Cnnxtavg    
    
    """
    
    #for k, fk  in enumerate(filename12):
    #    Sxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[:2*steps+1]
    #    Sxtavg += Sxtk
    #    Sx0t0 += Sxtk[0,0]
    
    #for k, fk  in enumerate(filename22):
    #    Nxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[:2*steps+1]
    #    Nxtavg += Nxtk
    #    Nx0t0 += Nxtk[0,0]
   
    """    

for L in L_:
    #this will only store the last iteration of the loop

    filename1, filename2, Cxt_, Cnnxt_= opencorr(L)
    steps, x_ = Cxt_.shape
    print(steps,x_)

    Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
    print('Cxtavg.shape = ', Cxtavg.shape)
    Cxtavg = Cxtavg/len(filename1)
    
    Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
    print('Cnnxtavg.shape = ', Cnnxtavg.shape)
    Cnnxtavg = Cnnxtavg/len(filename2)

Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)

#the whole point of slicing upto 128 steps
t = Cxtavg.shape[0]
t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array

#transforming x-axis from 0,L to -L/2, L/2
x = np.arange(x_) - int(0.5*x_)
X = np.arange(x_)

magstr = {}
magstr['magg'] =  'Magnetization'
magstr['stagg'] = 'Staggered magnetization'

filename = {}
filename['magg'] = filename1
filename['stagg'] = filename2

corr = {}
corr['magg'] = 'Cxt'
corr['stagg'] = 'Cnnxt'

#def calc_slope():
#    y_c = 

def plot_corrxt(magg):
    print('magg = ', magg)
    plt.figure()        
    fontlabel = {'family': 'serif', 'size': 16} #'weight', 'sytle', 'color'	
    fontlabel2 = {'family': 'serif', 'size': 12}
    fig, ax1 = plt.subplots(figsize=(10,10))
    #fig.suptitle(f'{magstr[magg]} correlations', fontsize=18, fontweight='bold')
   
    if magg == 'magg':
        corrxt_ = Cnewxt
    if magg == 'stagg':
        corrxt_ = Cnnnewxt
    
    #ax2 = plt.subplot(1,2,2)
    ax2 = ax1.inset_axes([0.1, 0.625, 0.25, 0.25])
    ax3 = ax1.inset_axes([0.675,0.625,0.25,0.25])
    
    p_opt, p_cov = optimization.curve_fit(func, np.arange(t), corrxt_[:,L//2])
    print('curve_fit params: ', p_opt, p_cov)
    #area_uc = []
    
    for ti in t_: # [0] 
        y = corrxt_[ti-1]
        auc = auc_simpson(X, y)
        print("area under the curve: ", auc)
        ax1.plot(x,y, label=f'{4*ti}')
        ax2.plot(x/ti**(0.5), ti**(0.5)*y, label=f'{4*ti}')
    dY_ = np.log(corrxt_[-1, L//2]/corrxt_[1, L//2])
    dX_ = np.log(np.arange(t)[-1]/np.arange(t)[1])
    print('slope of dcorrxt/dt: ' , dY_/dX_)
    ax3.loglog(np.arange(t)[1:], corrxt_[1:,L//2], '-.', linewidth=2)

    ax1.legend(); 
    #ax1.set_xlim(x[3*L//16], x[13*L//16+1])
    ax1.set_xlabel(r'$x$', fontdict=fontlabel); ax1.set_ylabel(r'$C(x,t) $',fontdict=fontlabel) ##C_{NN}(x,t)
    ax2.set_xlabel(r'$x/t^{0.5}$', fontdict=fontlabel2); ax2.set_ylabel(r'$t^{0.5}$C(x,t)', fontdict=fontlabel2)
    ax3.set_xlabel(r'$t$', fontdict=fontlabel2); ax3.set_ylabel(r'$C(0,t)$', fontdict=fontlabel2)
    
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_lambda_{}_mu_{}_{}_{}config.pdf'.format(corr[magg],max(L_),Lambda, Mu,epstr[epss],len(filename[magg])))
    #plt.show()

plot_corrxt('stagg')
plot_corrxt('magg')

