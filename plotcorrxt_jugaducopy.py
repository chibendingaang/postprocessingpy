#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
#a potential source of error could be at first line of opencorr(L, path) if the file under inspection spin_a_[123].dat is
#absent


import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import scipy.fft as sff
np.set_printoptions(threshold=2561)

L = int(sys.argv[1])
L_ = [L]
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb
dtstr = f'{dtsymb}emin3'    
fine_res = 4 #int(L/2048*10)
#if L = 2048: fine_res = 10; correspodning ratios to be picked

"""Inputs to the program:"""
Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])
begin = int(sys.argv[5])
end = int(sys.argv[6])

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 
choice = int(sys.argv[8])

epstr = dict()
epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}

"""
retrieves ~stores the output from obtainspinsnew into a file at given path
"""

if choice == 0:
    if Lambda == 1 and Mu == 0: param = 'xphsbg'
    if Lambda == 0 and Mu == 1: param = 'xpdrvn'
    if Lambda == 1 and Mu == 1: param = 'xpa2b0'
    path = f'./xpdrvn/L{L}/{dtstr}' 

if choice == 1:
    if Lambda == 1 and Mu == 0: param = 'qwhsbg'
    if Lambda == 0 and Mu == 1: param = 'qwdrvn'; path = f'./{param}/{dtstr}' #mostly
    if Lambda == 1 and Mu == 1: param = 'qwa2b0'
if choice == 2:
    if Lambda == 1 and Mu ==0: param = 'hsbg'
    if Lambda == 0 and Mu == 1: param = 'drvn'; path = f'./{param}/{dtstr}'
    if Lambda == 1 and Mu==1 : param = 'qwa2b0'

path0 = f'./L{L}/eps_min3/{param}'; #the epss = 3 for this path 
path1 = f'./{param}/{dtstr}'; #why differently heirarchical storage paths?
path2 = f'./xpdrvn/L{L}/{dtstr}' 

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

def auc_simpson(x,f):
    a = x[0]; b = x[-1]; n = len(x)
    h = (b-a)/(n-1)
    auc = (h/3)*(f[0]  + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])
    return auc

def func(t,a):
    return a*t**(-0.5) 
# make some trial data

def opencorr(L):
    Cxtpath = f'Cxt_storage/L{L}' #Cxt_storage
    Cxtpath1 = f'Cxt_series_storage/L{L}'
    #energypath = f'H_total_storage/L{L}'
    
    #filerepo = f'./{Cxtpath1}/outa_512.txt'
    if choice==1 and epss == 4:filerepo = f'./{Cxtpath1}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    filerepo = f'./{Cxtpath1}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    '''
    Note: L = 512 for the above filerepo: '...2emin3_emin3_out.txt' is not even showing diffusive curve
    '''
    #filerepo = f'./{Cxtpath}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    """biggest corrxt filerepos in the Cxt_storage/L512 directory:
    qwdrvn_cnnxt_2emin3_emin4_out.txt
    keep (epss, choice) = (4,1) for this folder; but avoid it as data is not smooth
    """
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()]) [begin:end]
    f.close()
    print(filename[::20]) 
    
    
    fk = filename[-1] #np.random.randint(0, len(filename))] 
    #find which file has smallest stepcount, if files are of non-uniform size
    Cnnxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)
    steps = int(Cnnxtk.size/L)
   
    print('~resolution~ smoothness between steps = ', fine_res, '\n #steps = ', steps) # , '\n stepcount = min(steps, 525)' )  
    steps = min(steps, 640) #pick fewer steps where necessary
    
    Cxtavg = np.zeros((steps,L)) #2*steps+1
    Cnnxtavg = np.zeros((steps ,L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
     
    
    for k, fk  in enumerate(filename):
        Cnnxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)[:steps]
        #keeping the same number of steps as Cxtavg;
        #Note: this may not work when: Cnnxtk.shape[0] < Cxtavg.shape[0]
        Cnnxtavg += Cnnxtk
    return Cnnxtavg, filename  #, Cnnxtavg , filename2 
    

""" loadcxt() should be inserted here if it is to be used """

def plot_corrxt(magg):
    print('magg = ', magg)
    plt.figure()        
    fontlabel = {'family': 'serif', 'size': 16} #'weight', 'style', 'color'	
    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,9))
    #fig.suptitle(f'{magstr[magg]} correlations', fontsize=18, fontweight='bold')
    ax2 = axes.inset_axes([0.1, 0.5, 0.3, 0.3])
    ax3 = axes.inset_axes([0.675,0.5,0.3,0.3])
    
    #for L in L_: 
        #we are using a single L!
        #this will only store the last iteration of the loop

        #filename1, filename2, 
    #the following conditions on epss are in no way complete.
    if epss == 3: path = path2;
    if epss == 4: path = path1;
    path = path1
    Cnnxt_, filenam2 =  opencorr(L) #,path) variable has become irrelevant: we only want the Cxtpath
    steps, x_ = Cnnxt_.shape
    print(steps,x_)
    
    Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1), :]
    print('C(nn)xtavg.shape = ', Cnnxtavg.shape)
    Cnnxtavg = Cnnxtavg/(end - begin) 
    Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)
    #Cnnxtavg is not sliced up symmetrically wrt x = 0; hence the concatenate command
    
    
    #the whole point of slicing upto 128 steps
    t0 = Cnnxtavg.shape[0]
    t = t0 - t0%32
    print('t0, t: ', t0, t)
    print('C(nn)xt shape: ', Cnnxtavg.shape)
    t_ = np.array([8, 12, 16, 24, 32, 48])
    #t_ = np.concatenate((np.arange(2*t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, t, t//4))) #t//16 in the first array
    #t_ = np.array([12, 16, 20 , 24])
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
    
    #if magg == 'magg':
    #    corrxt_ = loadcxt(L,path2)[0] 
    #if magg == 'stagg':
    #    corrxt_ = loadcxt(L,path2)[1]
    
    #important statement:
    corrxt_ = Cnnnewxt; #Cnewxt; 

    #ax4 = axes.inset_axes([0.625,0.05,0.25,0.25]) 
    #inset not needed for current purposes: ax4 is for a close-up view

    #p_opt, p_cov = optimization.curve_fit(func, np.arange(t), corrxt_[:,L//2])
    #print('curve_fit params: ', p_opt, p_cov)
    #area_uc = []
    
    for ti in t_: # [0] 
        y = corrxt_[ti-1]
        #y2 = Cnnnewxt[ti-1]
        #auc = auc_simpson(X, y)
        #print("area under the curve: ", auc)
        axes.plot(x,y, label=f'{10*ti}') #fine_res*
        #if ti<=80: 
        #    ax4.plot(x,y, label=f'{ti}') 
        #fine_res* : whether or not to multipy with fine_res is tricky;
        #tackle on a case-to-case basis
    for k in range(5):
        #auc_k = auc_simpson(fine_res*np.arange(t), np.abs(CMqt[:,k])/np.abs(CMqt[0,k]))
        #print(f"area under the curve: {auc_k} for k = {k}")
        #a constant area under the curve means diffusion spread is working
        ax2.plot(x/ti**(0.5), ti**(0.5)*y, label=f'{ti}') #instead of 4*ti since fine_res =5; 1/(dt*100) = 5; 5/fine_res = 1
   
    print('first, last entries of cnnxt(x=0): ', corrxt_[0, L//2], corrxt_[-1,L//2])
    print('first, last time steps considered: ', np.arange(t)[0], np.arange(t)[-1])
    dY_ = np.log(corrxt_[48, L//2]/corrxt_[0, L//2])
    dX_ = np.log(np.arange(t)[48]/1)
    print('slope of d(corrxt(x=0))/dt: ' , dY_/dX_)
    ax3.loglog(np.arange(t)[0:], corrxt_[1:,L//2], '-.', linewidth=2)

    axes.legend(); #ax4.legend()
    axes.set_xlim(-100, 100)
    ax2.set_xlim(x[5*L//16], x[11*L//16+1])
    ax3.set_xlim(1, 1000); ax3.set_ylim(0.001, 1)
    #ax4.set_xlim(-32,32); ax4.set_ylim(-0.125, 0.25)
    axes.set_xlabel(r'$x$',fontdict=fontlabel); axes.set_ylabel(r'$ C(x,t) $', fontdict=fontlabel) ##C_{NN}(x,t)
    #axes[1].set_xlabel(r'$x$',fontdict=fontlabel); axes[1].set_ylabel(r'$ CN(x,t) $', fontdict=fontlabel) ##C_{NN}(x,t)
    
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    fig.tight_layout()
    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_jump{}_{}config.png'.format(corr[magg],L, param,epstr[epss],fine_res, len(filename[magg])))
    #plt.show()

plot_corrxt('stagg')
#plot_corrxt('magg')


'''
load of unwanted shtatements
def loadcxt(L, path):
    for L in L_:
        #this will only store the last iteration of the loop

        #filename1, filename2, 
        Cnnxt_, filenam2  = opencorr(L, path1)
        steps, x_ = Cnnxt_.shape
        print('Total steps, sites: ', steps,x_)

        Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
        print('Cnnxtavg.shape = ', Cnnxtavg.shape)
        Cnnxtavg = Cnnxtavg/(end-begin) #len(filename2)

    Cnnnewxt = np.concatenate((Cnnxtavg[:,L//2:], Cnnxtavg[:,0:L//2]), axis = 1)

    #the whole point of slicing upto 128 steps
    t = Cnnxtavg.shape[0]; #L = Cxtavg.shape[1]
    t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array

    #transforming x-axis from 0,L to -L/2, L/2
    x = np.arange(x_) - int(0.5*x_)
    X = np.arange(x_)
    

    magstr = {}
    magstr['magg'] =  'Magnetization'
    magstr['stagg'] = 'Staggered magnetization'
    
    filename = {}
    filename['magg'] = filenam1
    filename['stagg'] = filenam2

    corr = {}
    corr['magg'] = 'Cxt'
    corr['stagg'] = 'Cnnxt'
    return Cnnnewxt #, CMqt , Cnnnewxt
'''
