#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
#a potential source of error could be at line#97 if the file under inspection spin_a_[123].dat is
#absent

#from calc_corrxt import calc_Cxt #,obtaincorrxt

from plot_corrxt import opencorr
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import scipy.fft as sff

#simulation parameters
L = int(sys.argv[1])
#L_ = [L, 512]
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb
dtstr = f'{dtsymb}emin3'    
fine_res = 5

Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])


#steps = (1280*L//1024)/dtsymb
#if dt==0.002: steps = 1280
#steps = int(L*0.05/dt)//32 

#configuration file indices
begin = int(sys.argv[5])
end = int(sys.argv[6])

np.set_printoptions(threshold=2561)

if Lambda == 1 and Mu == 0: param = 'xphsbg'
if Lambda == 0 and Mu == 1: param = 'xpdrvn'
if Lambda == 1 and Mu == 1: param = 'qwa2b0'

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss) 

epstr = dict()
epstr[3] =  'emin3'
epstr[4] =  'emin4' 
epstr[6] =  'emin6'
epstr[8] =  'emin8'

#some redundant labelling of files
epstr[33] = 'min3'
epstr[44] = 'min4'
choice = int(sys.argv[8])

#if choice ==0:
#    path = f'./xpdrvn/L{L}/{dtstr}'
#elif choice ==1:
#    path = './{}/{}'.format(param,dtstr)        #the epss = 3 for this path
#filerepo = f'{path}/outb.txt'

#Cxtpath = 'Cxt_storage/L{}'.format(L) #Cxt_storage
#energypath = f'H_total_storage/L{L}'

path0 = f'./L{L}/eps_min3/{param}'
#the epss = 3 for this path 
path1 = f'./{param}/{dtstr}'
path2 = f'./xpdrvn/L{L}/{dtstr}' 


def getcorr(L,path):
    """
    retrieves ~stores the output from obtainspinsnew into a file at given path
    """
    #Cxtpath = 'Cxt_storage/L{}'.format(L) #Cxt_storage
    #energypath = f'H_total_storage/L{L}'
    #conf = end+200
    # sample files labelled from config 32401 onwards are still saved, which give the file size description of all the configs used here
    Sp_aj = np.loadtxt(f'{path}/spin_a_{str(begin+32401)}.dat') 
    steps = int((Sp_aj.size/(3*L))) 
    print('steps = ', steps)
    
    Cxt = np.zeros((steps//fine_res,L)); Cnnxt = np.zeros((steps//fine_res,L))
 
    Sp_a = np.reshape(Sp_aj, (steps,L,3))[0:steps:fine_res]
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a*eta
    
    ''' measuring spin files at chosen interval for averaging
        this is to take interval sampling in the timeseries data and reduce the calculation time 
    '''
    Cxtavg = np.zeros((steps//fine_res,L)) #2*steps+1
    Cnnxtavg = np.zeros((steps//fine_res,L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
  
    #for k in range(begin,end): #enumerate(filename1):
        #Cxtk = calc_Cxt(Cxt,steps, Sp_a); 
        #Cnnxtk = calc_Cxt(Cnnxt,steps, Sp_ngva)
    
    '''this is it: we are just reading the existing Cxt and Cnnxt files through the opencorr() module,
       which sources these files from the path 
    '''
    Cxtk, Cnnxtk = opencorr(L, path) 
    #suppressing the Cnnxtk output since we seem to have lost some data pertaining to Cnnxt_*_qwdrvn* files, which needs to be sorted quickly
    Cxtavg += Cxtk; Cnnxtavg += Cnnxtk
    Cxtavg = np.array(Cxtavg)
    Cnnxtavg = np.array(Cnnxtavg)

    return Cxtavg[1:,:], Cnnxtavg[1:,:]
    #avoiding returning the 0th time step, which may be 0


#for conf in range(begin, end):
Cxt_, Cnnxt_ = getcorr(L, path1) 

#this will only store the last iteration of the loop
#filename1, filename2, Cxt_, Cnnxt_= opencorr(L)
#Cxt_, Cnnxt_= opencorr(L)

print('Cxtavg.shape: ', Cxt_.shape)

Cxt_avg = Cxt_/(end-begin)
Cnnxt_avg = Cnnxt_/(end-begin) 

steps, x_ = Cxt_.shape
print('Cxtavg.shape = ', Cxt_avg.shape)
#Cxtavg = Cxtavg/len(filename1)
   
'''
#the whole point of slicing upto 128 steps
t = Cxtavg.shape[0]
t_ = np.concatenate((np.arange(t//16,t//4,t//16), np.arange(3*t//8, t//2+1, t//8),np.arange(6*t//8, 7*t//8+1, t//4))) #t//16 in the first array

#transforming x-axis from 0,L to -L/2, L/2
x = np.arange(x_) - int(0.5*x_)
X = np.arange(x_)
'''
magstr = {}
magstr['magg'] =  'Magnetization'
magstr['stagg'] = 'Staggered magnetization'

#filename = {}
#filename['magg'] = filename1
#filename['stagg'] = filename2

corr = {}
corr['magg'] = 'Cxt'
corr['stagg'] = 'Cnnxt'

T, N = Cxt_avg.shape
print('T,N: ', T,N)
SMq_ = np.empty(Cxt_avg.shape)
SNq_ = np.empty(Cxt_avg.shape)

for ti in range(0,T):
    SMq_[ti] = 2*sff.fft(Cxt_avg[ti,:])/N
    SNq_[ti] = 2*sff.fft(Cnnxt_avg[ti,:])/N
    
print('SMq_ shape: ', SMq_.shape)

fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(9,9))

for k in range(3):
    ax1.plot(fine_res*np.arange(T), np.abs(SMq_[:,int(2**(k-1))]), label=f'k = {int(2**(k-1))}')
    #ax2.plot(fine_res*np.arange(T), np.abs(SNq_[:,int(2**(k-1))]), label=f'k = {int(2**(k-1))}')

ax1.set_xlabel('t'); #ax2.set_xlabel('t')
ax1.set_xlim(0,520)
#ax2.set_xlim(0,320)
#ax1.grid()
#ax2.grid()
ax1.set_title(r'$<\vec{M}(k,0) \vec{M}(-k,t)>$')
#ax2.set_title(r'$<\vec{N}(k,0) \vec{N}(-k,t)>$')
ax1.legend(); #ax2.legend()
#plt.suptitle(r'$<\vec{M}(k,0) \vec{M}(-k,t)>$')
plt.tight_layout()
plt.savefig('./plots/Cqt_vs_t_{}_{}config.png'.format( param,end-begin)) #len(filename['stagg'])))

"""
def plot_corrxt(magg):
    print('magg = ', magg)
    plt.figure()        
    #font = {'family': 'serif', 'size': 16} #'weight', 'sytle', 'color'	
    fig, ax1 = plt.subplots(figsize=(12,11))
    fig.suptitle(f'{magstr[magg]} correlations', fontsize=18, fontweight='bold')
   
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
    ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'C(x,t) $') ##C_{NN}(x,t)
    ax2.set_xlabel(r'$x/t^{0.5}$'); ax2.set_ylabel(r'$t^{0.5}$C(x,t)')
    ax3.set_xlabel(r't'); ax3.set_ylabel(r'C(0,t)')
    
    #plt.title(r'$\lambda = {}, \mu = {}$'.format(Lambda,Mu))
    plt.savefig('./plots/{}_vs_x_L{}_lambda_{}_mu_{}_{}_{}config.png'.format(corr[magg],max(L_),Lambda, Mu,epstr[epss],len(filename[magg])))
    #plt.show()

plot_corrxt('stagg')
plot_corrxt('magg')
"""
