#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
#a potential source of error could be at line#97 if the file under inspection spin_a_[123].dat is
#absent

#from calc_corrxt import obtaincorrxt, calc_Cxt
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import scipy.fft as sff

L = int(sys.argv[1])
#L_ = [L, 512]
#L_ = [L]
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
    
    Sp_a = np.reshape(Sp_aj, (steps,L,3))[0:steps:40]
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a*eta

    Cxtavg = np.zeros((16*(steps//640),L)) #2*steps+1
    Cnnxtavg = np.zeros((16*(steps//640),L))
    print('Cxtavg.shape: ' , Cxtavg.shape)
  
    #path = f'Cxt_storage/L{L}'
    path = f'Cxt_series_storage/L{L}' #Cxt_storage
    filerepo = f'./{path}/{param}_cxt_{dtstr}_{epstr[epss]}_out.txt'
    f = open(filerepo)
    filename1 = np.array([x for x in f.read().splitlines()]) [begin:end]
    f.close()
    
    filerepo = f'./{path}/{param}_cnnxt_{dtstr}_{epstr[epss]}_out.txt'
    g = open(filerepo)
    filename2 = np.array([x for x in g.read().splitlines()]) [begin:end]
    g.close()
    
    for k, fk  in enumerate(filename1):
        Cxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[1:16*(steps//640)+1]
        Cxtavg += Cxtk
    
    for k, fk  in enumerate(filename2):
        Cnnxtk = np.load('./{}/{}'.format(path, fk), allow_pickle=True)[1:16*(steps//640)+1]
        Cnnxtavg += Cnnxtk

    #for k in range(begin,end): #enumerate(filename1):
    #    Cxtk = calc_Cxt(Cxt,steps, Sp_a); #np.load(f'./{path}/{fk}', allow_pickle=True)[:16*(steps//640)+1]
    #    Cnnxtk = calc_Cxt(Cnnxt,steps, Sp_ngva)
    #    Cxtavg += Cxtk; Cnnxtavg += Cnnxtk

    #return Cxtavg, Cnnxtavg
    return filename1, filename2, Cxtavg, Cnnxtavg    

#for conf in range(begin, end):
#    Cxt, Cnnxt, CExt, CENxt, energ_1, energ_eta1  = obtaincorrxt(conf)
#    Cxt_avg += Cxt; Cnnxt_avg += Cnnxt

#this will only store the last iteration of the loop
filename1, filename2, Cxt_, Cnnxt_= opencorr(L)
#Cxt_, Cnnxt_= opencorr(L)

Cxt_avg = Cxt_/(end-begin)
Cnnxt_avg = Cnnxt_/(end-begin) 

steps, x_ = Cxt_.shape
#Cxtavg = Cxt_ #[:(16*(steps//16) + 1), :]
print('Cxtavg.shape = ', Cxt_avg.shape)
#Cxtavg = Cxtavg/len(filename1)
   
#Cnnxtavg = Cnnxt_ #[:(16*(steps//16) + 1),:]
#print('Cnnxtavg.shape = ', Cnnxt_avg.shape)
#Cnnxtavg = Cnnxtavg/len(filename2)

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

filename = {}
filename['magg'] = filename1
filename['stagg'] = filename2

corr = {}
corr['magg'] = 'Cxt'
corr['stagg'] = 'Cnnxt'

T, N = Cxt_avg.shape
print('T,N: ', T,N)
SMq_ = np.zeros(Cxt_avg.shape)
SNq_ = np.zeros(Cxt_avg.shape)

for ti in range(T):
    SMq_[ti] = 2*sff.fft(Cxt_avg[ti,:])/N
    SNq_[ti] = 2*sff.fft(Cnnxt_avg[ti,:])/N

 
print('SNq_ shape: ', SNq_.shape)
print('SMq_(k\'s) : ', np.abs(SMq_)[0], np.abs(SMq_)[1])

fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,figsize=(9,14))

for k in range(3):
    print('Timedecay of the k-th mode: ', np.where(np.abs(SMq_[:,int(2**(k-1))])/np.abs(SMq_[:,0]) <= np.exp(-1)))
    print('Timedecay of the k-th mode: ', np.where(np.abs(SNq_[:,int(2**(k-1))])/np.abs(SNq_[:,0]) <= np.exp(-1)))
    ax1.semilogy(np.arange(T), np.abs(SMq_[:,int(2**(k-1))]), label=f'MM corr; k = {int(2**k)}')
    ax2.semilogy(np.arange(T), np.abs(SNq_[:,int(2**(k-1))]), label=f'NN corr; k = {int(2**k)}')

ax1.set_xlabel('t'); ax2.set_xlabel('t')
ax1.set_ylabel(r'$C_{MM}(k,t)$')
ax2.set_ylabel(r'$C_{NN}(k,t)$')
ax1.set_xlim(0,80)
ax2.set_xlim(0,80)
#ax1.grid()
#ax2.grid()
#plt.suptitle(r'$<\vec{S}(k,0) \vec{S}(-k,t)>$')
ax1.legend(); ax2.legend()
plt.tight_layout()
plt.savefig('./plots/Strqt_vs_t_loglog_{}_{}config.png'.format(param, len(filename['stagg'])))

SMqw_ = np.zeros(SMq_.shape)
SNqw_ = np.zeros(SNq_.shape)

for ki in range(N):
    SMqw_[:,ki] = 2*sff.fft(SMq_[:,ki])/T
    SNqw_[:,ki] = 2*sff.fft(SNq_[:,ki])/T


fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,figsize=(9,14))
for wi in range(5):
    #print('Timedecay of the omega-th mode: ', np.where(np.abs(SMq_[wi,:])/np.abs(SMq_[0,:]) <= np.exp(-1)))
    #print('Timedecay of the omega-th mode: ', np.where(np.abs(SNq_[wi,:])/np.abs(SNq_[0,:]) <= np.exp(-1)))
    ax1.semilogy(2*np.arange(N)/N, np.abs(SMqw_[wi,:]), label=f'MM corr; w = {wi}')
    ax2.semilogy(2*np.arange(N)/N, np.abs(SNqw_[wi,:]), label=f'NN corr; w = {wi}')

ax1.set_xlabel('t'); ax2.set_xlabel('t')
ax1.set_ylabel(r'$C_{MM}(k,w)$')
ax2.set_ylabel(r'$C_{NN}(k,w)$')
#ax1.set_xlim(0,80)
#ax2.set_xlim(0,80)
#ax1.grid()
#ax2.grid()
#plt.suptitle(r'$<\vec{S}(k,0) \vec{S}(-k,t)>$')
ax1.legend(); ax2.legend()
plt.tight_layout()
plt.savefig('./plots/Powerspeqw_vs_t_loglog_{}_{}config.png'.format(param, len(filename['stagg'])))
