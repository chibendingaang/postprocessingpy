import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.fft as sff
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy import integrate
import scipy.optimize as optimization

L = int(sys.argv[1]) #512
dim3 = 3
steps = 1601
T, N = 1601, 512
X = np.arange(-L//2, L//2)

path = f'./L{L}/2emin3'
fine_r = 2 #int(sys.argv[1])
eps = 10 #10by100
j = np.random.randint(11,32)

'''
bin files
inp_file = f"{path}/spin_a_200{j}.bin"
spina = np.fromfile(inp_file, dtype=np.float64)
'''
if L==512: qmode_arr = [1,2,4] #,16]
if L==960: qmode_arr = [1,2,3,4,5,6]

'''
dat files
m = int(sys.argv[1])  
wavemode for files going as spin_a/b_modulated20(03) : m = 1 for 2000 series, m =2 for 3000 series... but m=8 >> 8100
'''

def openfile(m):
    #m = qmode_arr[0]
    inp_file = f"{path}/spin_a_pertbd{m+1}1{j}.dat" #modulated or pertbd 
    #the pertbd choice is taken considering the fact the we want N_tot(t=0) = 0
    #so that we do not have to deal with time varying values of N, however slow the variation is

    spina = np.loadtxt(inp_file, dtype=np.float64) #'{}/spin_a_210{}.dat'.format(path,j))
    sp_a = np.reshape(spina, (steps , L, dim3))
    #print('Configuration file: ', inp_file)
    #print(spina.shape)
    #this gives an array of length 3*L*N
    #N = number of time steps
    return sp_a


"""
def reshape_arr(spin,L):
    dim1 = int(spin.shape[0]/(L*3)) #,L,dim3)
    sp_ = np.reshape(spin, (dim1, L, dim3))
    return sp_
"""

def calc_dxt(sp_a, sp_b):
    decorr = np.sum(sp_a*sp_b, axis=2)
    dxt = np.concatenate((decorr[:,L//2:], decorr[:, 0:L//2]), axis=1)
    return dxt

def calc_etaenerg(k):
    sp_a = openfile(qmode_arr[k])
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a*eta
    Spnbr_ngva = Spnbr_a*eta 
    Eeta_loc = (-1)*np.sum(Sp_a*Spnbr_ngva,axis=2)
    return Eeta_loc

def calc_cxt(sp_a, steps):
    Cxt_ = np.zeros((steps//fine_r+1, L))    
    for ti,t in enumerate(range(0,steps,fine_r)):
        #print('t: ', t)
        for x in range(L):
            Cxt_[ti,x] = np.sum(np.sum((sp_a*np.roll(np.roll(sp_a,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(steps//fine_r+1 - ti)     #multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
    return Cxt_


def calc_avgmag(m,steps):
    """
    to plot the autocorrelator <M(t0).M(t+t0)>
    """
    num_slices = 1
    
    autocorr_m = np.zeros(steps)
    sp_a = openfile(m); print('m = ', m)
    #sp_a = np.reshape(spina, (steps , L, dim3)) #[0:steps:fine_r] 
    avgmag = np.sum(sp_a[:L//num_slices], axis=0)/L
    
    #roll is for window sliding
    #[-ti] is to take heavyside function of the correlator function since delta_t < 0 is not permissible
    for ti in range(0,steps):
        autocorr_m[ti] = np.sum(np.sum((avgmag*np.roll(avgmag, -ti,axis=0))[:steps-ti],axis=1))/(steps +1 - ti); 
        #two nested sums are needed else: putting sequence where scalar should be error shows up
    print('autocorr.shape: ', autocorr_m.shape)
    return avgmag, autocorr_m


def calc_mkt(m, steps):
    sp_a = openfile(m); #print('m = ', m)  #[0:steps:fine_r] 
    #steps = sp_a.shape[0]
    Sh = sp_a.shape; #print(Sh)
    Mq_1, Mq_2, Mq_3 = np.empty((Sh[0],Sh[1])), np.empty((Sh[0],Sh[1])), np.empty((Sh[0],Sh[1]))
    T, N = Mq_3.shape

    for ti in range(0,T):
        Mq_1[ti] = sff.fft(sp_a[ti,:,0])/N
        Mq_2[ti] = sff.fft(sp_a[ti,:,1])/N
        Mq_3[ti] = sff.fft(sp_a[ti,:,2])/N

    Mq_ = np.sqrt(np.abs(Mq_1)**2 + np.abs(Mq_2)**2 + np.abs(Mq_3)**2) 
    print('Mq_ shape: ', Mq_.shape) #shape: (steps, q-modes)

    #Mq_0 = Mq_[0]
    #print(np.abs(Mq_[0,m]))
    #print(np.abs(Mq_[200,m]))
    return Mq_, Mq_3


#Cxt_ = np.zeros((steps//fine_r+1, L))
#Cxt_ = calc_cxt(sp_a, steps)
#X = np.arange(-L//2, L//2)

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


def check_areauc(k,inputf):
    #Mq_, Mq_3 = calc_mkt(k, steps)
    T = np.arange(steps)
    func = inputf #[:160,k] #instead of Mq_
    T1 = T #[:160]
    #print('func.shape, T1.shape')
    #print(func.shape); print(T1.shape)
    f1 = CubicSpline(T1, func)(T1) 
    #putting the reference of the x-axis at the end is crucial; otherways it does not return an indexed array
    f2 = Akima1DInterpolator(T1, func)(T1)
    #f3 = PchipInterpolator(T1, func)(T1)
    print('f1/f2.shape: ', f1.shape, f2.shape) #, f3.shape)
    auc1 = integrate.simpson(f1,T1)
    auc2 = integrate.simpson(f2,T1)
    #auc3 = integrate.simpson(f3,T1)
    return f1, f2, auc1, auc2

#f1, f2, f3, auc1, auc2, auc3 = check_areauc(2)

def funq2(qarr, a, b):
    return a/(qarr)**b

cutoff = np.exp(-2)


def plot_mkt():
    plt.figure()
    fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(9,9))
    ax2 = ax1.inset_axes([0.65, 0.65, 0.275, 0.25])
    kthread = ""
    tau_q = []
    for k in qmode_arr[0:]: #range(0,m+1):
        avgmag, autocorr_m = calc_avgmag(k,steps) #shapes: (L,steps); (steps,)
        Mq_, Mq_3 = calc_mkt(k, steps)
        Mqprod = Mq_[:,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k])
        
        """
        we need to define an inputf that is as close to exp(-t/tau) as possible
        g(t) = f(t) - A; where is the long time value
        \tau = (\int_0^{\tau} g(t) dt )/g(0)
        so first we need to figure out the value A : average value of the function at long times
        """
        Mq_longtime = np.average(Mqprod[120:]) #the value of A
        #print('Mq_longtime: ', Mq_longtime)
        
        autocorrmq = sff.fft(autocorr_m)/steps
        
        inputg = autocorr_m #q #Mq_[:160,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k])
        inputg0 = inputg[0]
        
        #inputf = Mqprod - Mq_longtime #Mq_[:160,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k])
        #inputf0 = inputf[0]
        #print('inputg.shape: ', inputg.shape)
        #f1, f2,  auc1, auc2 = check_areauc(k, inputf)
        f1, f2,  auc1, auc2 = check_areauc(k, inputg)
        auc1 = auc1/inputg0; auc2 = auc2/inputg0
        tau_q.append(auc2)
        kthread += str(k)
        
        #T, N = Mq_3.shape
        T = autocorrmq.shape
        
        #boolean array masking whether Mq(t,qin) < cutoff*Mq(0,qin)
        #maskMq = np.abs(Mq_[:,k]) < cutoff*np.abs(Mq_[0,k])
        #print(threeconsecutivetrues(maskMq))
        print('AUCs from different fittings: ', auc1, auc2 ) 
         
        ax1.plot(np.arange(steps), autocorr_m, label='k = {k}') #fine_r * np.arange(T)
        #ax2.plot(np.arange(T), Mq_[:,k]*Mq_[0,k]/(Mq_[0,k]*Mq_[0,k]), label=f'M(t,k)M(0,k) ; k = {k}') #fine_r * np.arange(T) 
        #ax2.plot(np.arange(T), np.abs(SMq_2[:,int(2**(k-1))]), label=f'MM corr; inputq = 2; k = {int(2**k)}     
        ax1.set_xlabel('t'); #ax2.set_xlabel('t');
        ax1.set_xlim(0,1600); #ax2.set_ylim(0,0.6)
        ax2.plot(np.arange(steps)[:100], autocorr_m[:100])
    
    plt.suptitle(r'$ \langle M(t_0).M(t+t_0) \rangle $'); ax1.legend(); # ax2.legend(); plt.tight_layout(); 
    plt.savefig('./plots/AutocorrMqt_modltdq{}_{}th.pdf'.format( kthread,j)) #len(filename['stagg']))
    #plt.savefig('./plots/SMqt_vs_t_drvn_ordered_modltdq{}_{}th.pdf'.format( m,j)) #len(filename['stagg']))) 
    
    tau_q = np.array(tau_q)
    f_tauq = CubicSpline(qmode_arr, tau_q)
    new_qs = np.arange(1, 17, 1)
    pred_tauq = f_tauq(new_qs)
    print(pred_tauq.shape)
    plt.figure()
    plt.plot(qmode_arr, tau_q, 'o', label = 'data')
    plt.plot(new_qs, pred_tauq, '-', label = 'interpolate')
    
    
    
    p_opt, p_cov = optimization.curve_fit(funq2, new_qs, pred_tauq)
    print('popt, pcov: ', p_opt, p_cov)
    inv_q2 = p_opt[0]/new_qs**p_opt[1]
    plt.plot(new_qs, inv_q2, '-.', label ='{}/q^{}'.format(round(p_opt[0],2), round(p_opt[1],2))) #1/(new_qs)**2, '-.', label = '1/q^2')
    plt.legend()
    plt.title('Decay time v/s q')
    plt.savefig('./plots/autocorrTauq_{}_{}th.pdf'.format(kthread,j))
    
    plt.figure()
    plt.loglog(new_qs, inv_q2, '-.', label ='{}/q^{}'.format(round(p_opt[0],2), round(p_opt[1],2)))
    plt.title('Decay time v/s q')
    plt.savefig('./plots/autocorrlogTauq_{}_{}th.pdf'.format(kthread,j))
    
    plt.figure()
    plt.plot(qmode_arr, 1/tau_q )
    plt.title('inverse decay time v/s q')
    plt.savefig('./plots/autocorrinvTauq_{}_{}th.pdf'.format(kthread,j))

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
