import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.fft as sff
from scipy.interpolate import CubicSpline, Akima1DInterpolator 
from scipy import integrate
import scipy.optimize as optmz
import scipy.signal as sig

#from plot_tauk_corrkt import threeconsecutivetrues #, check_auc
path_ = "/home/nisarg/entropy_production/mpi_dynamik/xpdrvn/L512/2emin3"

L = 512
dt = 0.002
dtsym = 2; fine_res = 10
param = 'drvn'
qmode_arr = [1,2,4,6,8,10,12,14,16] #,10,12,14,16,18,20,24,28,32]
#qmode = 2
#step1: load spin_array from given filename

def loadfile(qmode):
    Sp_raw = np.loadtxt(f'{path_}/spin_a_modltd_q{qmode}.dat')

    #reshape array according to L,steps,n parameters
    steps = int((Sp_raw.size/(3*L)))    
    Sp_a = np.reshape(Sp_raw, (steps, L, 3))

    og_shape = Sp_a.shape
    stepcount = min(steps, 800)
    #this many steps are enough to satisfy our constraints 
    Sp_a = Sp_a[:stepcount, :, :]
    print('Sp_a.shape, stepcount: ', og_shape, ', ', stepcount)
    T = np.arange(stepcount)

    return Sp_a, stepcount, T

#Sp_raw = loadfile(qmode)

def check_x_intercept(func):
    x_intercepts = []
    for i, fi in enumerate(func[0:-1]):
        if fi*func[i+1] < 0: x_intercepts.append(i)
    return x_intercepts

def threeconsecutiveones(arr):
    count = 0
    for i in range(len(arr)):
        if arr[i] == 1: count += 1
        else: count = 0
        if count >= 3:
            return i-2; break

def localextrema(arr):
    loc_max = sig.argrelextrema(arr, np.greater)
    loc_min = sig.argrelextrema(arr, np.less)
    return loc_max, loc_min

#def tau_mthd3(func):
    #the steady state method:
    #use the tau_methods 1 and 2 to get the tau_relaxation
    #then find the steady state tau through the correlator



def tau_mthd2(func):
    '''
    this method involves using localextrema to mask the input func
    and then checking if the three consecutive maxima are decreasing
    returns: the indices of the local maxima and boolean array: (func[t_max] <= cutoff * func[0])
    '''

    print("Sp_3[t=0]: ", func[0])
    #knowing beforehand what our input func will be
    cutoff = 0.1 #np.exp(-4) 
    print("cutoff*Sp3[0] = ", cutoff*func[0])
    
    loc_max, loc_min = localextrema(func)
    loc_max = np.array(loc_max).flatten() 
    mask_cutoff = np.abs(func[loc_max]) < cutoff*np.abs(func[0])
    #or simply use np.abs(func[loc_max[0]]) instead of flatten
    mask_cutoff = mask_cutoff*1
    taus_ = T[loc_max]*mask_cutoff
    tau_T = taus_[np.where(taus_ != 0)[0][0]]
    #this is the function that requires both the criteria to be True
    print(' loc_max, mask_cutoff : ', loc_max, mask_cutoff)
    return loc_max, func[loc_max], mask_cutoff, tau_T


def tau_mthd1(func):
    #this function interpolates input function func to f1 through CubicSpline
    #two methods for tau: 
    
    #method1: just find five consecutive x-intersections, and ensure that 3 consecutive peaks are reducing
    #         the local peaks lie between 0 and x1, x2 and x3, x4 and x5
    #to find x-intercept: check when f1(x1).f2(x2) <0 and take x_intercept = (x1+x2)*0.5
    f = CubicSpline(T, func)(T)    
    t_intercepts = check_x_intercept(f)
    if len(t_intercepts) >= 4:
        tau = t_intercepts[4]
        return T[tau]
    else: 
        print('Tau cannot be found since function damped out quickly')


def spinfft(spin,stepcount):
    sp1q = np.zeros(spin.shape[:2])
    sp2q = np.zeros(spin.shape[:2])
    sp3q = np.zeros(spin.shape[:2])
    for ti in range(spin.shape[0]):
        sp1q[ti] = 2*sff.fft(spin[ti,:,0])/L
        sp2q[ti] = 2*sff.fft(spin[ti,:,1])/L
        sp3q[ti] = 2*sff.fft(spin[ti,:,2])/L
    return sp1q, sp2q, sp3q

xf = 2*np.pi*sff.fftfreq(L)
X = np.arange(-L//2, L//2)

def check_areauc(k,inputf):
    #Mq_, Mq_3 = calc_mkt(k, steps)
    T = np.arange(stepcount)
    func = inputf[:stepcount]
    #only 0th mode of S3(q,t) to be considered, we need a 1-D array for splinfitting
    #[:160,k] #instead of Mq_
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


fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(12,10))
#plt.figure(figsize=(9,8))
font = {'family':'serif', 'size':16}
font2 = {'family':'serif', 'size':14}
qthread = ''
tauq1 = []
tauq2 = []
Tauq_1 = []
Tauq_2 = []
for qmode in qmode_arr:
    Sp_a, stepcount, T = loadfile(qmode)
    sp1q, sp2q, sp3q = spinfft(Sp_a, stepcount)
    sp3_q0 = sp3q[:,0]
    Sp_abs = np.sqrt(sp1q**2 + sp2q**2 + sp3q**2)
    Mqprod = Sp_abs[:,qmode]*Sp_abs[0,qmode]/(Sp_abs[0,qmode]*Sp_abs[0,qmode])
    #use this correlation function before the decay; and in the stedy state as well
    
    Sp_0 = Sp_abs[:, 0]
    f1, f2, tau1, tau2 = check_areauc(qmode, Sp_0)
    tauq1.append(tau2)
    f1, f2, tau1, tau2 = check_areauc(qmode, Mqprod)
    Tauq_1.append(tau2)

    loc_max, sp3q_maxloc, mask_cutoff, tau_T = tau_mthd2(Sp_0)
    #loc_tau = threeconsecutiveones(mask_cutoff)
    #sp3q_maxloc[loc_tau] 
    tauq2.append(tau_T)
    
    loc_max, sp3q_maxloc, mask_cutoff, tau_T = tau_mthd2(Sp_0)
    Tauq_2.append(tau_T)
    
    qthread += str(qmode)
    #axes.plot(T, sp1q[:,qmode], label=rf'$S^{(1)}(q=$ {qmode})')
    #axes.plot(T, sp2q[:,qmode], label=rf'$S^{(2)}(q=$ {qmode})')
    axes[0].plot(T[:320], Sp_abs[:320,0], label=rf'$k_i, ${qmode}')
    axes[1].plot(T[:320], sp3_q0[:320], label=rf'$k_i, ${qmode}')
    #axes.plot(T, np.abs(Sp_abs[:,qmode]), label=rf'|S(q,t).S(q,t)|)')
    #since each component is <0.5 in magnitude, Sp_abs < Sp_3
tauq1 = np.array(tauq1)
tauq2 = np.array(tauq2)
Tauq_1 = np.array(Tauq_1)
Tauq_2 = np.array(Tauq_2)

print(tauq1, tauq2)
print(Tauq_1, Tauq_2)

ax2 = axes[0].inset_axes([0.65, 0.65, 0.25, 0.25])
ax2.plot(qmode_arr, tauq1, label='AUC')
ax2.plot(qmode_arr, tauq2, label='cutoff+intercept')
ax2.set_xlabel('k', fontdict=font2)
ax2.set_ylabel(r'$\tau$', fontdict=font2)

#ax3 = axes[1].inset_axes([0.65, 0.65, 0.25, 0.25])
#ax3.plot(qmode_arr, tauq1, label='AUC')
#ax3.plot(qmode_arr, tauq2, label='cutoff+intercept')
#ax3.set_xlabel('k', fontdict=font2)
#ax3.set_ylabel(r'$\tau$', fontdict=font2)

axes[0].set_xlabel('t',fontdict=font)
axes[0].set_ylabel(r'$|S(q,t)|$',fontdict=font)
axes[0].legend()


axes[1].set_xlabel('t',fontdict=font)
axes[1].set_ylabel(r'$S(q,t)*S(q,0)$',fontdict=font)
axes[1].legend()

fig.tight_layout()

plt.savefig(f'./plots/Mkt_k_{qthread}.png')	
