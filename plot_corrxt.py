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
import scipy.fft as sff

L = int(sys.argv[1])
#L_ = [L, 512]
L_ = [L]
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb
dtstr = f'{dtsymb}emin3'    
fine_res = 20

#steps = (1280*L//1024)/dtsymb
#if dt==0.002: dtstr = '2emin3'; steps = 1280
#int(L*0.05/dt)//32 

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])


epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 
choice = int(sys.argv[8])

np.set_printoptions(threshold=2561)

if choice == 0:
    if Lambda == 1 and Mu == 0: param = 'xphsbg'
    if Lambda == 0 and Mu == 1: param = 'xpdrvn'
    if Lambda == 1 and Mu == 1: param = 'xpa2b0'
if choice == 1:
    if Lambda == 1 and Mu == 0: param = 'qwhsbg'
    if Lambda == 0 and Mu == 1: param = 'qwdrvn'
    if Lambda == 1 and Mu == 1: param = 'qwa2b0'

epstr = dict()
epstr[3] =  'emin3'
epstr[4] =  'emin4' 
epstr[6] =  'emin6'
epstr[8] =  'emin8'

epstr[33] = 'min3'
epstr[44] = 'min4'

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


"""
retrieves ~stores the output from obtainspinsnew into a file at given path
"""

#Cxtpath = f'Cxt_series_storage/L{L}' #Cxt_storage
#energypath = f'H_total_storage/L{L}'

path0 = f'./L{L}/eps_min3/{param}'; 
#the epss = 3 for this path 
path1 = f'./{param}/{dtstr}'; 
path2 = f'./xpdrvn/L{L}/{dtstr}' 

def opencorr(L,path):
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(begin+32401))) #why 32400 filenumber
    #demofile to open and check array length: steps, L, 3
    steps = min(int((Sp_aj.size/(3*L))), 525) 
    print('steps = ', steps)
    print('stepcount = min(steps, 525)')
    
    Cxtavg = np.zeros((steps//fine_res,L)) #2*steps+1
    Cnnxtavg = np.zeros((steps//fine_res ,L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
 
    Cxtpath = f'Cxt_series_storage/L{L}' #Cxt_storage
    #path = f'Cxt_storage/L{L}'

    filerepo = f'./{Cxtpath}/{param}_cxt_{dtstr}_{epstr[epss]}_out.txt'
    #filerepo = f'./{Cxtpath}/thefilerepo1_{param}.txt'
    f = open(filerepo)
    filename1 = np.array([name for name in f.read().splitlines()]) [begin:end]
    f.close()
    print(filename1[::100]) 
    
    for k, fk  in enumerate(filename1):
        Cxtk = np.load(f'./{Cxtpath}/{fk}', allow_pickle=True)[:steps//fine_res]
        Cxtavg += Cxtk
    print('Cxtk.shape: ', Cxtk.shape)

    filerepo = f'./{Cxtpath}/outcnn_qwdrvn.txt'
    #'{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt': out of the 3092 files here, only 201 Cnnxt files remain!
    g = open(filerepo)
    filename2 = np.array([x for x in g.read().splitlines()]) [begin:end]
    g.close()
    print(filename2[::100]) 
    
    for k, fk  in enumerate(filename2):
        Cnnxtk = np.load(f'./{Cxtpath}/{fk}', allow_pickle=True)[:steps//fine_res]
        Cnnxtavg += Cnnxtk
    return Cxtavg, filename1 , Cnnxtavg , filename2 
    #here the Cxtavg isn't really averaged, it is the summation over all the configs 
    
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


##commenting temporarily; for the next 40 lines
###

def loadcxt(L, path):
    for L in L_:
        #this will only store the last iteration of the loop

        #filename1, filename2, 
        Cxt_, filenam1 , Cnnxt_ , filenam2 = opencorr(L, path1)
        steps, x_ = Cxt_.shape
        print(steps,x_)

        Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
        print('Cxtavg.shape = ', Cxtavg.shape)
        Cxtavg = Cxtavg/(end - begin) #len(filename1)
    
        Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
        print('Cnnxtavg.shape = ', Cnnxtavg.shape)
        Cnnxtavg = Cnnxtavg/(end-begin) #len(filename2)

    Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
    Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)

    #the whole point of slicing upto 128 steps
    t = Cxtavg.shape[0]; #L = Cxtavg.shape[1]
    t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array

    #transforming x-axis from 0,L to -L/2, L/2
    x = np.arange(x_) - int(0.5*x_)
    X = np.arange(x_)
    
    CMqt = np.zeros(Cxtavg.shape) 
    for ti in range(0,t):
        CMqt = 2*sff.fft(Cxtavg[ti,:])/t

    magstr = {}
    magstr['magg'] =  'Magnetization'
    magstr['stagg'] = 'Staggered magnetization'
    
    filename = {}
    filename['magg'] = filenam1
    filename['stagg'] = filenam2

    corr = {}
    corr['magg'] = 'Cxt'
    corr['stagg'] = 'Cnnxt'
    return Cnewxt, CMqt , Cnnnewxt

#def calc_slope():
#    y_c = 

def plot_corrxt(magg):
    print('magg = ', magg)
    plt.figure()        
    fontlabel = {'family': 'serif', 'size': 16} #'weight', 'style', 'color'	
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(13,9))
    #fig.suptitle(f'{magstr[magg]} correlations', fontsize=18, fontweight='bold')
   
    for L in L_:
        #this will only store the last iteration of the loop

        #filename1, filename2, 
        Cxt_, filenam1, Cnnxt_, filenam2 = opencorr(L, path1)
        steps, x_ = Cxt_.shape
        print(steps,x_)

        Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
        print('Cxtavg.shape = ', Cxtavg.shape)
        Cxtavg = Cxtavg/(end - begin) #len(filename1)
    
        Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
        print('Cnnxtavg.shape = ', Cnnxtavg.shape)
        Cnnxtavg = Cnnxtavg/(end-begin) #len(filename2)

    Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
    Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)

    #the whole point of slicing upto 128 steps
    t = Cxtavg.shape[0]
    print('Cxt shape: ', Cxtavg.shape)
    t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array

    #transforming x-axis from 0,L to -L/2, L/2
    x = np.arange(x_) - int(0.5*x_)
    X = np.arange(x_)

    CMqt = np.zeros(Cxtavg.shape) ; CNqt = np.zeros(Cnnxtavg.shape)
    for ti in range(0,t):
        CMqt[ti] = 2*sff.fft(Cxtavg[ti,:])/t
    print('Cmqt shape: ', CMqt.shape)
 
    magstr = {}
    magstr['magg'] =  'Magnetization'
    magstr['stagg'] = 'Staggered magnetization'
    
    filename = {}
    filename['magg'] = filenam1
    filename['stagg'] = filenam2

    corr = {}
    corr['magg'] = 'Cxt'
    corr['stagg'] = 'Cnnxt'
    
    #if magg == 'magg':
    #    corrxt_ = loadcxt(L,path2)[0] 
    #if magg == 'stagg':
    #    corrxt_ = loadcxt(L,path2)[1]
    
    #important statement:
    corrxt_ = Cnewxt; #Cxtavg
     
    #ax2 = plt.subplot(1,2,2)
    #ax2 = ax1.inset_axes([0.1, 0.625, 0.25, 0.25])
    #ax3 = ax1.inset_axes([0.6,0.625,0.25,0.25])
    ax4 = axes[0].inset_axes([0.625,0.05,0.25,0.25])
    ax5 = axes[1].inset_axes([0.625,0.05,0.25,0.25])
    #p_opt, p_cov = optimization.curve_fit(func, np.arange(t), corrxt_[:,L//2])
    #print('curve_fit params: ', p_opt, p_cov)
    #area_uc = []
    
    for ti in t_: # [0] 
        y = corrxt_[ti-1]
        y2 = Cnnnewxt[ti-1]
        #auc = auc_simpson(X, y)
        #print("area under the curve: ", auc)
        axes[0].plot(x,y, label=f'{fine_res*ti}')
        axes[1].plot(x,y2, label=f'{fine_res*ti}')
        if ti<=80: 
            ax4.plot(x,y, label=f'{fine_res*ti}')
            ax5.plot(x,y2, label=f'{fine_res*ti}')
    '''
    for k in range(5):
        axes[1].plot(
        #axes1.plot(fine_res*np.arange(t), np.abs(CMqt[:,k]), label=f'k={k}')
        auc_k = auc_simpson(fine_res*np.arange(t), np.abs(CMqt[:,k])/np.abs(CMqt[0,k]))
        print(f"area under the curve: {auc_k} for k = {k}")
        #ax2.plot(x/ti**(0.5), ti**(0.5)*y, label=f'{ti}') #instead of 4*ti since fine_res =5; 1/(dt*100) = 5; 5/fine_res = 1
        if (fine_res*ti%40 ==0): ax4.plot(x,y, label=f'{fine_res*ti}')
    '''
    dY_ = np.log(corrxt_[-1, L//2]/corrxt_[1, L//2])
    dX_ = np.log(np.arange(t)[-1]/np.arange(t)[1])
    print('slope of dcorrxt/dt: ' , dY_/dX_)
    #ax3.loglog(np.arange(t)[1:], corrxt_[1:,L//2], '-.', linewidth=2)

    axes[1].legend(); axes[1].legend(); #ax4.legend()
    #ax1.set_xlim(x[3*L//16], x[13*L//16+1])
    ax4.set_xlim(-32,32); ax4.set_ylim(-0.125, 0.25)
    ax5.set_xlim(-32,32); ax5.set_ylim(-0.125, 0.25)
    axes[0].set_xlabel(r'$x$',fontdict=fontlabel); axes[0].set_ylabel(r'$ CM(x,t) $', fontdict=fontlabel) ##C_{NN}(x,t)
    axes[1].set_xlabel(r'$x$',fontdict=fontlabel); axes[1].set_ylabel(r'$ CN(x,t) $', fontdict=fontlabel) ##C_{NN}(x,t)
    #ax2.set_xlabel(r'$x/t^{0.5}$',fontdict=fontlabel); ax2.set_ylabel(r'$t^{0.5} C(x,t) $',fontdict=fontlabel)
    #ax3.set_xlabel(r't',fontdict=fontlabel); ax3.set_ylabel(r'$C(0,t)$', fontdict=fontlabel)
    
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_lambda_{}_mu_{}_{}_{}config.pdf'.format(corr[magg],max(L_),Lambda, Mu,epstr[epss],len(filename[magg])))
    #plt.show()

plot_corrxt('stagg')
#plot_corrxt('magg')

