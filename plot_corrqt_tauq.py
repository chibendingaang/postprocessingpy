import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.fft as sff
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy import integrate
import scipy.optimize as optimization


L = int(sys.argv[1])
path = f'./L{L}/2emin3'
fine_r = 2 #int(sys.argv[1])
eps = 10 #10by100
#j = np.random.randint(10,32)
t_0 = int(sys.argv[2 ]) #300
t_max = 1600 #int(sys.argv[2]) #200, 320, 500, 800, 
#ensure that t_max >=400; after which the results start to get consistent for the Cmqt_stt

qthread = ''
'''
bin files
inp_file = f"{path}/spin_a_200{j}.bin"
spina = np.fromfile(inp_file, dtype=np.float64)
'''
#keep even number of entries in the array
qmode_arr = [1,2,3,4,5,6] #,10,12,14,16] #,16]

'''
dat files
m = int(sys.argv[1])  
wavemode for files going as spin_a/b_modulated20(03) : m = 1 for 2000 series, m =2 for 3000 series... but m=8 >> 8100
'''

def openfile(m):
    inp_file = f'{path}/spin_a_modltd_q{m}.dat'
    #inp_file = f"{path}/spin_a_modulated{m+1}1{j}.dat" #except when m is 8; check input params
    spina = np.loadtxt(inp_file, dtype=np.float64) #'{}/spin_a_210{}.dat'.format(path,j))
    #print('Configuration file: ', inp_file)
    print('Spin_arr.shape: ', spina.shape)
    #this gives an array of length 3*L*N
    #N = number of time steps
    return spina

L = 512; dim3 = 3
steps = 1601
T, N = 1601, 512
X = np.arange(-L//2, L//2)
Xf = 2*np.pi*sff.fftfreq(L)

"""
#note the formula below, for steady-state averaging
def calc_cxt(sp_a, steps):
    Cxt_ = np.zeros(steps//fine_r+1, L)    
    for ti,t in enumerate(range(0,steps,fine_r)):
        print('t: ', t)
        for x in range(L):
            Cxt_[ti,x] = np.sum(np.sum((sp_a*np.roll(np.roll(sp_a,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(steps//fine_r+1 - ti)     #multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
    return Cxt_
"""


def threeconsecutivetrues(arr):
    count = 0
    #convert the boolean array to 1's and 0's if not already done
    arr = arr*1
    for i in range(len(arr)):
        if arr[i] == 1: count = count +1
        else: count = 0
        if count >= 3: return i-2

#print(maskMq)
#print(np.where(np.abs(Mq_[:,m])/np.abs(Mq_[0,m]) < np.exp(-2)))


def check_areauc(inputf):
    T = np.arange(steps)
    func = inputf[:-1100] #[:t_max,k] #at some point it depended on k, not anymore
    #inputf instead of Mq_
    T1 = T[t_0:t_max-1100] # or till T[:t_0]
    #print('func.shape, T1.shape')
    #print(func.shape); print(T1.shape)
    f1 = CubicSpline(T1, func)(T1) 
    
    #putting the reference of the x-axis at the end is crucial; otherways it does not return an indexed array
    print('f1.shape: ', f1.shape) #, f3.shape)
    auc1 = integrate.simpson(f1,T1)
    return f1, auc1

#f1, f2, f3, auc1, auc2, auc3 = check_areauc(2)



def calc_mkt(m, steps):
    spina = openfile(m); 
    print('m = ', m)
    sp_a = np.reshape(spina, (steps , L, dim3)) #[0:steps:fine_r] 
    #steps = sp_a.shape[0]
    Sh = sp_a.shape; 
    Mq_1, Mq_2, Mq_3 = np.empty((Sh[0],Sh[1])), np.empty((Sh[0],Sh[1])), np.empty((Sh[0],Sh[1]))
    T, N = Mq_3.shape
    
    #Fourier Transform of spatial component
    for ti in range(0,T):
        Mq_1[ti] = sff.fft(sp_a[ti,:,0])/N
        Mq_2[ti] = sff.fft(sp_a[ti,:,1])/N
        Mq_3[ti] = sff.fft(sp_a[ti,:,2])/N

    Mq_ = np.sqrt((Mq_1)**2 + (Mq_2)**2 + (Mq_3)**2) #avoid np.abs() over each Mq_i component
    print('Mq_ shape: ', Mq_.shape) #shape: (steps, q-modes)

    #Mq_0 = Mq_[0]
    print(f'|M(q,0)|  : ', np.abs(Mq_[0,m]))
    print(f'|M(q,t0)| : ', np.abs(Mq_[t_0,m]))
    return Mq_, Mq_1, Mq_2, Mq_3, sp_a

mthread = ''
fontlabel = {'family': 'Serif', 'size':16}

"""
fig0, axes0 = plt.subplots(nrows =1, ncols =1, figsize=(9,8)) #(11,8_
#axnew2 = axes[1].inset_axes([0.6,0.15, 0.28,0.28])

for m in qmode_arr:
    Mq_, Mq_1, Mq_2, Mq_3, sp_a = calc_mkt(m,steps)
    
    #Sp_alt = np.sum(np.sum(sp_a**2, axis=2),axis=1)/L #single L here for avg due to the order of summation over components first
    Sp_tot = (np.sum(sp_a[:,:,0],axis=1)**2 + np.sum(sp_a[:,:,1], axis=1)**2 + np.sum(sp_a[:,:,2],axis=1)**2)/L**2
    #this is = (Mtot_1)^2 + (Mtot_2)^2 + (Mtot_3)^2
    print('Sp_.shape: ' , Sp_tot.shape)
    Mq_tot = (np.sum(Mq_1**2,axis=1) + np.sum(Mq_2**2, axis=1) + np.sum(Mq_3**2, axis=1))
    print('Mq_.shape: ', Mq_tot.shape)
    
    axes0.plot(np.arange(steps)[:t_0+200], Sp_tot[:t_0+200], label=f'k_i= {m}')
    #axes[1].plot(np.arange(steps)[:t_0+200], Mq_tot[:t_0+200], label=f'k_i={m}, k = 0') #instead of Mq_[:,0]
    mthread += str(m)
'''
m = 2
axnew2.plot(Xf, Mq_[t_0,:], label=f't= {t_0}, k = {m}')
axnew2.plot(Xf, Mq_[t_max,:], label=f't= {t_max}, k = {m}')    
m = 8
axnew2.plot(Xf, Mq_[t_0,:], label=f't= {t_0}, k = {m}')
axnew2.plot(Xf, Mq_[t_max,:], label=f't= {t_max}, k = {m}')    
axnew2.set_xlabel('k', fontdict=fontlabel)
axnew2.set_ylabel(f'M(k,t)',fontdict=fontlabel)

axnew2.legend()
'''
axes0.set_xlabel('t', fontdict=fontlabel)
axes0.set_ylabel(r'$\langle M^2(t) \rangle_x $',fontdict=fontlabel)
axes0.set_ylim(-0.005,1.005);
#axes[1].set_ylim(-0.005,1.005)
##axes[1].set_xlabel('t', fontdict=fontlabel)
#axes[1].set_ylabel(r'$\langle M^2(t) \rangle_k $',fontdict=fontlabel)

axes0.legend()
#axes[1].legend()

fig0.tight_layout()
fig0.savefig(f'./plots/Mt_t_{t_max}minus{t_0}_q{mthread}.pdf') #_Mktabs_
"""

def funq2(qarr, a, b):
    return a/(qarr)**b

cutoff = 0.1 #np.exp(-2)
fontlabel = {'family': 'Serif', 'size':16}
Xf = 2*np.pi*sff.fftfreq(L)


def stdyst_cxt(k):
    qmodes_ = [1,2,3,4]
    Cmqt_stt = np.zeros((t_max - t_0, len(qmodes_)))
    Cm_q0t_stt = np.zeros((t_max - t_0)) 
    tau_q = []
    for qn, q in enumerate(qmodes_): 
        #k is the input perturbation mode
        #q are the q-dependent modes
        Mq_, Mq_1, Mq_2, Mq_3, sp_a = calc_mkt(k, steps)
        Mq1_stt  = Mq_1[t_0:]
        Mq2_stt  = Mq_2[t_0:]
        Mq3_stt  = Mq_3[t_0:]
        #Mq1_ltavg = np.average(Mq1_stt[:t_max]); Mq2_ltavg = np.average(Mq2_stt[:t_max]); Mq3_ltavg = np.average(Mq3_stt[:t_max])
        #Mq_prod = (Mq1_stt - Mq1_ltavg)*(Mq1_stt - Mq1_ltavg) + (Mq2_stt - Mq2_ltavg)*(Mq2_stt - Mq2_ltavg) + (Mq3_stt - Mq3_ltavg)*(Mq3_stt - Mq3_ltavg)
        #Mqprod_stt = np.zeros(Mq_stt.shape)
        
        for tn, t in enumerate(range(0, t_max-t_0)):
            Cmqt_stt[tn,qn] = np.sum(Mq1_stt[:,qn]*np.roll(Mq1_stt[:,qn],-tn) + Mq2_stt[:,qn]*np.roll(Mq2_stt[:,qn], -tn) + Mq3_stt[:,qn]*np.roll(Mq3_stt[:,qn], -tn))/(t_max - (t_0 + tn))
            Cm_q0t_stt[tn] = np.sum(np.sum(sp_a[:,:,0],axis=1)*np.roll(np.sum(sp_a[:,:,0],axis=1),-tn) +np.sum(sp_a[:,:,0],axis=1)*np.roll(np.sum(sp_a[:,:,0],axis=1),-tn) +np.sum(sp_a[:,:,0],axis=1)*np.roll(np.sum(sp_a[:,:,0],axis=1),-tn))/(N**2*(t_max-(t_0+tn)))
                               #np.sum(np.sum(Mq1_stt,axis=1)*np.roll(np.sum(Mq1_stt,axis=1),-tn) /
                               # + np.sum(Mq2_stt,axis=1)*np.roll(np.sum(Mq2_stt,axis=1), -tn) /
                               # + np.sum(Mq3_stt,axis=1)*np.roll(np.sum(Mq3_stt,axis=1), -tn))/(N**2*(t_max - (t_0 + tn)))
            
            #Cmqt_stt[tn,kn] = np.sum(Mq_prod*np.roll(Mq_prod,-tn,axis=0))/(t_max - (t_0 +t))
        inputf = Cmqt_stt[:,qn]
        f1, auc1 = check_areauc(inputf)
        auc1 = auc1 #/inputf0;
        tau_q.append(auc1)
        
    return Cmqt_stt, Cm_q0t_stt, np.array(tau_q), np.array(qmodes_)




def plot_mkt():
    ncolumns = len(qmode_arr)//2 
    fig, axes = plt.subplots(nrows=2, ncols=ncolumns,figsize=(11,8)) # (ax1,.. # (11,8)
    fig3, axes3 = plt.subplots(nrows=2, ncols=ncolumns,figsize=(11,8)) # (ax1,.. # (11,8)
    kthread = ""  
    #k2th = int(np.log2(kth)) #//2**kth
    for kn,k in enumerate(qmode_arr): #[k2th:k2th+1]: 
        Cmqt_stt, Cm_q0t_stt, tau_q, qmodes_ = stdyst_cxt(k)
        #Mq_, Mq_1, Mq_2, Mq_3, Sp_a = calc_mkt(k, steps)
        #this is not needed again
        #boolean array masking whether Mq(t,qin) < cutoff*Mq(0,qin)
        #maskMq = np.abs(Mq_[:,k]) < cutoff*np.abs(Mq_[0,k])
        #print(threeconsecutivetrues(maskMq))
        
        #print("AUC from inputf: ", auc1) 
        qmode_array = np.array(qmode_arr)
        '''
        print('tau_q.shape, qmode_arr.shape: ', tau_q.shape, qmode_array.shape)
        f_tauq = CubicSpline(qmode_array, tau_q)
        new_qs = np.arange(1, qmode_array[-1]+1, 1)
        pred_tauq = f_tauq(new_qs)
        #print(pred_tauq.shape)
        '''
        
        
        #Mqprod = (Mq_1[:,k]*Mq_1[0,k] + Mq_2[:,k]*Mq_2[:,k] + Mq_3[:,k]*Mq_3[:,k]) #/(Mq_[0,k]*Mq_[0,k]) #Mq_[:,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k])
        """
        we need to define an inputf that is as close to exp(-t/tau) as possible
        g(t) = f(t) - A; where is the long time value
        \tau = (\int_0^{\tau} g(t) dt )/g(0)
        so first we need to figure out the value A : average value of the function at long times
        """
        
        '''
        Mq_ltavg = np.average(Mqprod[t_0:]) #the value of A
        print('Mq_ltavg: ', Mq_ltavg)
        inputf = Mqprod - Mq_ltavg 
        inputf0 = inputf[0]
        print('inputf.shape: ', inputf.shape)
        '''
 
        kthread += str(k)
        #T, N = Mq_3.shape
         
        #ax1.plot(np.arange(T)[:t_max], Mq_[:t_max,k], label=f'k = {k}') #fine_r * np.arange(T)
        #axes[0,kn%ncolumns].plot(np.arange(T)[:t_max], Mq_[:t_max,0], label=f'k_i = {k}') #fine_r * np.arange(T)
        axes[kn//ncolumns,kn%ncolumns].plot(np.arange(T)[0:t_max-1100-t_0], Cm_q0t_stt[:-1100] , label=rf'q = 0')
        axes[kn//ncolumns,kn%ncolumns].plot(np.arange(T)[0:t_max-1100-t_0], Cmqt_stt[:-1100,0] , label=rf'q = 1')
        axes[kn//ncolumns,kn%ncolumns].plot(np.arange(T)[0:t_max-1100-t_0], Cmqt_stt[:-1100,1] , label=rf'q = 2')
        axes[kn//ncolumns,kn%ncolumns].plot(np.arange(T)[0:t_max-1100-t_0], Cmqt_stt[:-1100,2] , label=rf'q = 3')
        axes[kn//ncolumns,kn%ncolumns].plot(np.arange(T)[0:t_max-1100-t_0], Cmqt_stt[:-1100,3] , label=rf'q = 4')   
        """ having trial-plotted the following interpolations they fit precisely
        ax1.plot(np.arange(T)[:160], f1, '-.', label=f'CubicSpline; k = {k}') 
        ax1.plot(np.arange(T)[:160], f2, '--', label=f'Akima1DInterpolator; k = {k}') """
        #old: ax2.plot(np.arange(T)[:t_max], Mqprod[:t_max] , label=f'M(t,k)M(0,k) ; k = {k}') #Mq_[:,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k]);  #fine_r * np.arange(T) 
        #ax2.plot(np.arange(T)[t_0:t_max-500], Cmqt_stt[:-500,kn] , label=f'k_i = {k}') #Mq_[:,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k]);  #fine_r * np.arange(T) 
        #toggle with different k's and different (tmax - t)
        
        #plt.figure()
        axes3[kn//ncolumns,kn%ncolumns].plot(qmodes_, tau_q, '-o') #, label = 'data')
        #plt.plot(new_qs, pred_tauq, '-', label = 'interpolate')
    
        
        axes[kn//ncolumns,kn%ncolumns].set_xlabel('t0'); #ax2.set_xlabel('t');#ax3.set_xlabel('t')
        axes[kn//ncolumns,kn%ncolumns].set_ylabel(r'$\langle M(q,t) \cdot M(-q,t+t0) \rangle_t$'); 
        axes[kn//ncolumns,kn%ncolumns].set_title(rf'$k_i = $ {k}')
        axes3[kn//ncolumns,kn%ncolumns].set_xlabel('q')
        axes3[kn//ncolumns,kn%ncolumns].set_ylabel(r'$\tau$')
        axes[kn//ncolumns,kn%ncolumns].set_title(rf'$k_i = $ {k}')
        #ax2.set_ylabel(r'$\langle \frac{M(k,t) \cdot M(k,0)}{|M(k,0)|^2} \rangle_k$')
        #ax2.set_ylabel(r'$\langle M(q,t_0)\cdot M(-q,t+t_0) \rangle_t0$')
        axes[kn//ncolumns,kn%ncolumns].legend()
        axes3[kn//ncolumns,kn%ncolumns].set_ylim(-0.0025, 0.005)
        #axes[1,2].legend()
    #Mq_centralized = np.concatenate(Mq_[:,0:Mq_.shape[1]//2]
    #plt.suptitle(r'$|M(k,t)|$ and $\frac{M(q,t) \cdot M(-q,0)}{|M(k,0)|^2} $'); ax1.legend(); ax2.legend(); 
    #ax1.legend(); ax2.legend();
    fig.tight_layout(); 
    fig3.tight_layout()
    fig.suptitle(rf'$t_i = $ {t_0}')
    fig3.suptitle(rf'$t_i = $ {t_0}')
    #plt.savefig('./plots/Mqt_modltdq{}_{}th.pdf'.format( kthread,j)) 
    fig.savefig('./plots/Cmq_q0t_single_modltdq{}_{}minus{}_.pdf'.format( kthread,t_max, t_0))  
    fig3.savefig('./plots/Tauq_{}_{}minus{}.pdf'.format(kthread,t_max, t_0))
    #fig, axis = plt.subplots(figsize=(8,8)) 
    #axis.plot(Xf, Mq_[t_0,:]); #dont write Xf  - it gives a horizontal line 
    #plt.savefig(f'./plots/Mkt_k_fft_{kthread}.pdf')

    '''
    Check how tau varies with q for each initial k.
    '''
               
    
    #p_opt, p_cov = optimization.curve_fit(funq2, new_qs, pred_tauq)
    #print('popt, pcov: ', p_opt, p_cov) #no fitting is needed since we aren't expecting a specific function class; interpolation
    #plt.legend()
    #plt.title('Decay time v/s q')
    #plt.savefig('./plots/Tauq_{}_{}minus{}.pdf'.format(kthread,t_max, t_0))

    
    '''
    plt.figure()
    plt.loglog(new_qs, inv_q2, '-.', label ='{}/q^{}'.format(round(p_opt[0],2), round(p_opt[1],2)))
    plt.title('Decay time v/s q')
    plt.savefig('./plots/logTauq_{}.pdf'.format(kthread))
    
    plt.figure()
    plt.plot(qmode_arr, 1/tau_q )
    plt.title('inverse decay time v/s q')
    plt.savefig('./plots/invTauq_{}.pdf'.format(kthread))
    '''

"""
def plot_cxt():
    plt.figure(figsize=(10,10))
    plt.plot(X, Cxt_)
    plt.xlabel('x')
    plt.ylabel(r'$C(x,t)$')
    #plt.title(r'$\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu)) #$D(x,t)$ heat map; $t$ v/s $x$;
    #plt.savefig('./plots/Cxt_L{}_Lambda_{}_Mu_{}_dt_{}_ordered.png'.format(L,Lambda,Mu,dtstr, len(filename)))
    #plt.show()

def plot_dxt():
    plt.figure(figsize=(10,10))
    plt.pcolormesh(dxt[:], cmap = 'seismic',vmin =  0, vmax = 1.0);
    plt.xlabel('x')
    plt.ylabel(r'$t$')
    #plt.title(r'$\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu)) #$D(x,t)$ heat map; $t$ v/s $x$;
    plt.colorbar()
    #plt.savefig('./plots/Dxt_L{}_Lambda_{}_Mu_{}_dt_{}_{}confg.png'.format(L,Lambda,Mu,dtstr, len(filename)))
    #plt.show()
"""

plot_mkt()
#plot_mkt(8)
#plot_mkt(16)
