#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First version on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sff

#params
L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb
dtstr = f'{dtsymb}emin3'    
Lambda = int(sys.argv[3])
Mu = int(sys.argv[4])

interval = 1
if dtsymb ==2: fine_res = 10
if dtsymb ==1: fine_res = 20

epss = int(sys.argv[5])
epsilon = 10**(-1.0*epss)
choice = int(sys.argv[6])

np.set_printoptions(threshold=2561)
#length of the array

#dictionary names
epstr = dict()
epstr[3] = 'eps_min3'
epstr[4] = 'eps_min4'
epstr[6] = 'eps_min6'
epstr[8] = 'eps_min8' 

if choice ==0:
    if Lambda == 1 and Mu == 0: param = 'xphsbg'
    if Lambda == 0 and Mu == 1: param = 'xpdrvn'
    if Lambda == 1 and Mu == 1: param = 'xpa2b0'
    path =  f'./{param}/L{L}/2emin3' #path0 #usually xpdrvn 
if choice ==1:
    if Lambda == 1 and Mu == 0: param = 'qwhsbg'
    if Lambda == 0 and Mu == 1: param = 'qwdrvn'
    if Lambda == 1 and Mu == 1: param = 'qwa2b0'
    path = f'./L{L}/{epstr[epss]}/{param}' #path1  #the largest file data is currently here: 17042023
#usually qwdrvn #the epss = 3 for this path
if choice ==2:
    if Lambda == 1 and Mu == 0: param = 'hsbg'
    if Lambda == 0 and Mu == 1: param = 'drvn'
    if Lambda == 1 and Mu == 1: param = 'qwa2b0'
    path = f'./{param}/{dtstr}'	#path2 # the param is drvn or hsbg here; choice = 2

qmode_arr = [1,2,4,8] 
start  = time.perf_counter()

def calc_autoCxt(autoCxt_, steps, spin):
    for ti,t in enumerate(range(0,steps,fine_res)):
        print('time: ', t)
        for x in range(L):
            autoCxt_[ti,x] = np.sum(spin[0,:,:]*np.roll(spin,-x,axis=1),axis=2)  	
            #multiplying an array Sxt[t=ti] with a constant vector Sxt[t=0,L=0]
    return autoCxt_/L


"""
 obstainspinxt():
 returns: 
 Cxt- the single configuration spin/magnetization (time-)correlator
 Cnnxt- single configuraiton staggered magnetization (time-)correlator
 energ_1- system energy at time
 energ_eta1- eta-based system energy at time
"""

def obtainspinxt(path,qmode): 
    Sp_aj = np.loadtxt('{}/spin_a_modltd_q{}.dat'.format(path,qmode))
    Sp_aj = np.loadtxt('{}/spin_a_pertbd_q{}.dat'.format(path,qmode))
    steps = int((Sp_aj.size/(3*L))) #0.1*dtsymb multiplication omitted
     
    #reshaped to count spin_xt at every fine_resolved time step
    Sp_a = np.reshape(Sp_aj, (steps,L,3)) #[0:steps:fine_res]
    stepcount = min(steps, 525)
    Sp_a = Sp_a[:stepcount] 
    print(f'steps = min({stepcount},{steps})')
    print('Sp_a.shape: ' , Sp_a.shape)
    return Sp_a, stepcount
    
def spinfft(spin, stepcount):
    #fourier transforms of the three spin components
    Sp1q = np.zeros(spin.shape[:2]) #(dim1,dim2) shape of spin array
    Sp2q = np.zeros(spin.shape[:2]) 
    Sp3q = np.zeros(spin.shape[:2]) 

    for ti in range(spin.shape[0]):
        Sp1q[ti] = sff.fft(spin[ti,:,0])/L
        Sp2q[ti] = sff.fft(spin[ti,:,1])/L
        Sp3q[ti] = sff.fft(spin[ti,:,2])/L
    return Sp1q, Sp2q, Sp3q

Xf = 2*np.pi*sff.fftfreq(L)
#fig1, axesfew = plt.subplots(nrows = 3, ncols = 1, figsize = (8.1,10.8))

fig2, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (9.6,8))
#ax1 = axesfew[0].inset_axes([0.075, 0.625, 0.325, 0.275])
#ax2 = axesfew[1].inset_axes([0.075, 0.625, 0.325, 0.275])
#ax3 = axesfew[2].inset_axes([0.075, 0.625, 0.325, 0.275])
ax4 = axes.inset_axes([0.625,0.5,0.25,0.25])
qthread = ''
tthread = ''

for qi, qval in enumerate(qmode_arr):
    qthread += str(qval)
    Sp_a, stepcount = obtainspinxt(path,qval)
    print("S1 shape: ", Sp_a.shape[:2])
    T = np.arange(stepcount)
    Sp1q, Sp2q, Sp3q = spinfft(Sp_a, stepcount)
    Spq_sq = Sp1q**2 + Sp2q**2 + Sp3q**2
    print('S1q shape: ', Sp1q.shape)
    
    '''   
 #for t in np.arange(0,300,50): 
    #    axesfew[0].plot(Xf, np.abs(Sp1q[t,:]), label=f't={t}'); 
    #    axesfew[1].plot(Xf, np.abs(Sp2q[t,:]), label=f't={t}');
    #    axesfew[2].plot(Xf, np.abs(Sp3q[t,:]), label=f't={t}'); 
    #    ax1.plot(Xf, np.abs(Sp1q[t,:])); ax2.plot(Xf, np.abs(Sp2q[t,:])); ax3.plot(Xf, np.abs(Sp3q[t,:]))
    #axesfew[0].legend()
    #axesfew[1].legend()
    #axesfew[2].legend()
    #ax1.set_xlim(-0.125, 0.125)
    #ax2.set_xlim(-0.125, 0.125)
    #ax3.set_xlim(-0.125, 0.125)
    #ax1.set_title(rf'$|S^{(1)}|$') 
    #ax2.set_title(rf'$|S^{(2)}|$') 
    #ax3.set_title(rf'$|S^{(3)}|$') 
    #axesfew[0].set_title(rf'$|S^{(1)}|$, $q=${qval}')
    #axesfew[1].set_title(rf'$|S^{(2)}|$, $q=${qval}')
    #axesfew[2].set_title(rf'$|S^{(3)}|$, $q=${qval}')
    
    
    #axes[qi,0].plot(T, Sp1q[:,qval], label=rf'$S^1, q= ${qval}'); 
    #axes[qi,0].plot(T, Sp2q[:,qval], label=rf'$S^2, q= ${qval}'); 
    #axes[qi,0].plot(T, Sp3q[:,qval], label=rf'$S^3, q= ${qval}'); #axes[qi,0].legend() 
    '''

    #axes.plot(T, Sp3q[:,0], label=rf'$S^3, q= 0 $'); #axes[qi,0].legend() 

    axes.plot(T, Spq_sq[:,0], label=rf'$q_i = {qval} $'); #axes[qi,0].legend() 
    ax4.plot(T, Sp3q[:,0], label=rf'$q_i = {qval} $')
    #axes[qi,1].plot(T, np.abs(Sp1q[:,qval]), label=rf'$|S^{(1)}|$, q={qval}'); 
    #axes[qi,1].plot(T, np.abs(Sp2q[:,qval]), label=rf'$|S^{(2)}|$, q={qval}'); 
    #axes[qi,1].plot(T, np.abs(Sp3q[:,qval]), label=rf'$|S^{(3)}|$, q={qval}'); #axes[qi,0].legend() 
    axes.legend(); #ax4.legend()
    #axes[qi,1].legend()
    axes.set_xlabel('t'); #axes.set_ylabel(r'$Re[S^{(3)}(t)]$')
    axes.set_ylabel(rf'$S^2(0,t)$')
    ax4.set_xlabel('t')
    ax4.set_ylabel(rf'$ Re[S_3 (0,t)]$')
    #axes[qi,1].set_xlabel('t'); axes[qi,1].set_ylabel(r'$|S^{(i)}(t)|$')
    #axes[qi,1].plot(T, Sp2q[:,qval], label=f'S(2), q={qval}'); axes[qi,1].legend()
    #axes[qi,1].set_xlabel('t'); axes[qi,1].set_ylabel(r'$Re[S^{(2)}(t)]$')
    #axes[qi,2].plot(T, Sp3q[:,0], label=f'S(3), q=0'); axes[qi,2].legend()
    #axes[qi,2].set_xlabel('t'); axes[qi,2].set_ylabel(r'$Re[S^{(3)}(t)]$')
    #plt.suptitle('one of them is q+1, one is qth mode')
    #axesfew[0].set_xlabel('q')
    #axesfew[1].set_xlabel('q')
    #axesfew[2].set_xlabel('q')
plt.tight_layout()
plt.savefig(f'./plots/Spinq_vs_t_modltd_q{qthread}.pdf')
#plt.savefig(f'./plots/Spin3q_absSpinq_vs_t_pertbd_q{qthread}.pdf')


def calc(qmode):
    Sp_a, stepcount = obtainspinxt(path,qmode)
    r = int(1./dt)
    #j = file_j
    autoCxt = np.zeros((stepcount//fine_res+1, L))
    autoCnnxt = np.zeros((stepcount//fine_res+1, L))
    Spnbr_a = np.roll(Sp_a,-1,axis=1) 		
    
    E_loc = (-1)*np.sum(Sp_a*Spnbr_a,axis=2) #energy at each site
    energ_1 = np.sum(E_loc, axis = 1) 	#attached the (-1) factor
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a*eta
    Spnbr_ngva = Spnbr_a*eta 
    Eeta_loc = (-1)*np.sum(Sp_a*Spnbr_ngva,axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis = 1) 	#attached (-1) factor
    
    autoCxt = calc_autoCxt(autoCxt,stepcount, Sp_a); autoCnnxt = calc_autoCxt(autoCnnxt,stepcount, Sp_ngva)
    return autoCxt[1:], autoCnnxt[1:], energ_1, energ_eta1
     
Cxtpath = 'Cxt_series_storage/L{}'.format(L)
'''
for conf in range(begin, end):
print(conf)
autoCxt, autoCnnxt, energ_1, energ_eta1  = obtaincorrxt(conf, path)

#easier condition: if Cxt (is not None): save the qwhsbg files; if Cnnxt(is not None): save the qwdrvn files;
f = open(f'./{Cxtpath}/Cxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, Cxt); f.close()
f = open(f'./{Cxtpath}/Cnnxt_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(f, Cnnxt); f.close()

g = open(f'./{energypath}/H_tot_t_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_1); g.close()
g = open(f'./{energypath}/H_tot_etat_{dtstr}_{epstr[epss]}_{param}_{conf}to{conf+1}config.npy','wb'); np.save(g, energ_eta1); g.close()
'''
print('processing time = ', time.perf_counter() - start)


