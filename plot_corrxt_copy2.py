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
#begin = int(sys.argv[5])
#end = int(sys.argv[6])


epss = int(sys.argv[5])
epsilon = 10**(-1.0*epss) 
choice = int(sys.argv[6])


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

"""
#this function is not needed
def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    #np.where(condition, [x,y]) -> x if condition True; y if False
    #eqv to np.nonzero(arr) -> True for nonzero elements
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]
"""

#again, use readymade scipy function scipy.integrate() for this purpose;

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
#filenums for eps_min3 data: L512/2emin3
#filenum_rollcall = [1005, 1310,1410,1510] #1100 and 1000 series give the same plots

#filenums for eps_min4 data: L1024/eps_min4
filenum_rollcall = [23005, 27010] #,1410,1510] #1100 and 1000 series give the same plots

#Cxtpath = f'Cxt_series_storage/L{L}' #Cxt_storage
#energypath = f'H_total_storage/L{L}'

path0 = f'./L{L}/eps_min3/{param}'; 
#the epss = 3 for this path 
path1 = f'./{param}/{dtstr}'; 
path2 = f'./xpdrvn/L{L}/{dtstr}' 

def opencorr(L,path,filenum):
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(filenum)))
    steps = min(int((Sp_aj.size/(3*L))), 525) 
    print('steps = ', steps)
    
    Cxtavg = np.zeros((steps//fine_res,L)) #2*steps+1
    Cnnxtavg = np.zeros((steps//fine_res ,L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
 
    Cxtpath = f'Cxt_storage/L{L}' #Cxt_(series_)storage
    #path = f'Cxt_storage/L{L}'

    #filerepo = f'./{Cxtpath}/{param}_cxt_{dtstr}_{epstr[epss]}_out.txt'
    filerepo = f'./{Cxtpath}/qwdrvn_cnnxt_2emin3_emin4_out.txt' #qwdrvn_cnnxt_2emin3_emin3_out.txt in Cxt_series_storage
    #thefilerepo1_{param}.txt'
    #f = open(filerepo)
    filename1 = np.array([f'Cnnxt_t_2emin3_eps_min4_qwdrvn_{filenum}to{filenum+1}config.npy'])
    #Cxt_t_2emin3_eps_min3_xpdrvn_{filenum}to{filenum+1}config.npy'])
    #f.close()
    print(filename1[::100]) 
    
    for k, fk  in enumerate(filename1):
        Cxtk = np.load(f'./{Cxtpath}/{fk}', allow_pickle=True)[:steps//fine_res]
        Cxtavg += Cxtk
    print('Cxtk.shape: ', Cxtk.shape)

    filerepo = f'./{Cxtpath}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    g = open(filerepo)
    filename2 = np.array([x for x in g.read().splitlines()]) [begin:end]
    g.close()
    print(filename2[::100]) 
    
    for k, fk  in enumerate(filename2):
        Cnnxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[:steps//fine_res]
        Cnnxtavg += Cnnxtk
    return Cnnxtavg, filename2 #, filename2 #, Cnnxtavg   
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
        Cnnxt_, filenam2 = opencorr(L, path2)
        steps, x_ = Cnnxt_.shape
        print(steps,x_)

        #Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
        #print('Cxtavg.shape = ', Cxtavg.shape)
        #Cxtavg = Cxtavg/(end - begin) #len(filename1)
    
        Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
        print('Cnnxtavg.shape = ', Cnnxtavg.shape)
        Cnnxtavg = Cnnxtavg/(end-begin) #len(filename2)

    Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
    #Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)

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
    #filename['magg'] = filenam1
    filename['stagg'] = filenam2

    corr = {}
    corr['magg'] = 'Cxt'
    corr['stagg'] = 'Cnnxt'
    return Cnnnewxt #, Cnnnewxt

#def calc_slope():
#    y_c = 

def plot_corrxt(magg):
    magstr = {}
    magstr['magg'] =  'Magnetization'
    magstr['stagg'] = 'Staggered magnetization'
    

    corr = {}
    corr['magg'] = 'Cxt'
    corr['stagg'] = 'Cnnxt'
    
    #if magg == 'magg':
    #    corrxt_ = loadcxt(L,path2)[0] 
    #if magg == 'stagg':
    #    corrxt_ = loadcxt(L,path2)[1]
    print('magg = ', magg)
    
    plt.figure()        
    fontlabel = {'family': 'serif', 'size': 16} #'weight', 'sytle', 'color'	
    fontlabel2 = {'family':'serif', 'size':12}
    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    #fig.suptitle(f'{magstr[magg]} correlations', fontsize=18, fontweight='bold')
   
    
    for filen,filenum in enumerate(filenum_rollcall):
        #this will only store the last iteration of the loop

        #filename1, filename2, 
        Cnnxt_, filenam2 = opencorr(L, path2,filenum)
        steps, x_ = Cnnxt_.shape
        print(steps,x_)

        #Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
        #print('Cxtavg.shape = ', Cxtavg.shape)
        #Cxtavg = Cxtavg #/(end - begin) #len(filename1)
    
        Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
        print('Cnnxtavg.shape = ', Cnnxtavg.shape)
        Cnnxtavg = Cnnxtavg/(end-begin) #len(filename2)

        #Cnewxt = np.concatenate((Cxtavg[:,L//2:], Cxtavg[:,0:L//2]), axis = 1)
        Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)

        #the whole point of slicing upto 128 steps
        t = Cxtavg.shape[0]
        t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array

        #transforming x-axis from 0,L to -L/2, L/2
        x = np.arange(x_) - int(0.5*x_)
        X = np.arange(x_)

        #important statement:
        corrxt_ = Cnnnewxt; #Cxtavg
     
        #ax2 = plt.subplot(1,2,2)
        #ax2 = ax1.inset_axes([0.1, 0.625, 0.25, 0.25])
        #ax3 = ax1.inset_axes([0.6,0.625,0.25,0.25])
        ax1 = axes[filen//2, filen%2]
        ax4 = ax1.inset_axes([0.1,0.625,0.25,0.25])
    
        #p_opt, p_cov = optimization.curve_fit(func, np.arange(t), corrxt_[:,L//2])
        #print('curve_fit params: ', p_opt, p_cov)
        #area_uc = []
        
        #plot at specific times from the array    
        for ti in t_: # [0] 
            y = corrxt_[ti-1]
            #auc = auc_simpson(X, y)
            #print("area under the curve: ", auc)
            axes[(filen)//2,filen%2].plot(x,y, label=f'{fine_res*ti}')
            #ax2.plot(x/ti**(0.5), ti**(0.5)*y, label=f'{ti}') #instead of 4*ti since fine_res =5; 1/(dt*100) = 5; 5/fine_res = 1
            if (fine_res*ti%40 ==0): ax4.plot(x,y, label=f'{fine_res*ti}')
        dY_ = np.log(corrxt_[-1, L//2]/corrxt_[1, L//2])
        dX_ = np.log(np.arange(t)[-1]/np.arange(t)[1])
        print('slope of dcorrxt/dt: ' , dY_/dX_)
    #ax3.loglog(np.arange(t)[1:], corrxt_[1:,L//2], '-.', linewidth=2)

        ax1.legend(); ax4.legend() 
        #ax1.set_xlim(x[3*L//16], x[13*L//16+1])
        ax4.set_xlim(-32,32); ax4.set_ylim(-0.125, 0.25)
        ax1.set_xlabel(r'$x$',fontdict=fontlabel); ax1.set_ylabel(r'$ C(x,t) $',fontdict=fontlabel) ##C_{NN}(x,t)
        ax4.set_xlabel(r'$x/t^{0.5}$',fontdict=fontlabel2); ax4.set_ylabel(r'$ C(x,t) \cdot  t^{0.5} $',fontdict=fontlabel2) ##C_{NN}(x,t)
    #ax2.set_xlabel(r'$x/t^{0.5}$'); ax2.set_ylabel(r'$t^{0.5} C(x,t) $')
    #ax3.set_xlabel(r't'); ax3.set_ylabel(r'$C(0,t)$')
    
    filename = {}
    #filename['magg'] = filenam1
    filename['stagg'] = filenam2
    fig.tight_layout()
    plt.savefig('./plots/{}multiplots_vs_x_L{}_lambda_{}_mu_{}_{}_{}config.pdf'.format(corr[stagg],max(L_),Lambda, Mu,epstr[epss],len(filename[stagg])))
    #plt.show()

#plot_corrxt('stagg')
plot_corrxt('stagg')

