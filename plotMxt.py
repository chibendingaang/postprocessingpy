//nimport numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.fft as sff

path = '.'
fine_r = 2 #int(sys.argv[1])
eps = 10 #10by100
j = np.random.randint(10,32)

#bin files
#inp_file = f"{path}/spin_a_200{j}.bin"
#spina = np.fromfile(inp_file, dtype=np.float64)

#dat files
m = int(sys.argv[1])  #wavemode for files going as spin_a/b_modulated20(03) : m = 1 for 2000 series, m =2 for 3000 series...
inp_file = f"{path}/spin_a_modulated20{j}.dat"
spina = np.loadtxt(inp_file, dtype=np.float64) #'{}/spin_a_210{}.dat'.format(path,j))

print('Configuration file: ', inp_file)
print(spina.shape)
#this gives an array of length 3*L*N
#N = number of time steps

L = 512; dim3 = 3
steps = 1601

def reshape_arr(spin,L):
    dim1 = int(spin.shape[0]/(L*3)) #,L,dim3)
    sp_ = np.reshape(spin, (dim1, L, dim3))
    return sp_

sp_a = np.reshape(spina, (steps , L, 3)) #[0:steps:fine_r] 
#sp_a = reshape_arr(spina, L)
##sp_b = reshape_arr(spinb, L)
steps = sp_a.shape[0]

plt.plot(np.arange(L),sp_a[0,:,0])
plt.plot(np.arange(L),sp_a[0,:,2])
plt.savefig('abc3.pdf')

"""
def calc_dxt(sp_a, sp_b):
    decorr = np.sum(sp_a*sp_b, axis=2)
    dxt = np.concatenate((decorr[:,L//2:], decorr[:, 0:L//2]), axis=1)
    return dxt

#Cxt_ = np.zeros((steps//fine_r+1, L))

def calc_cxt(sp_a, steps):
    for ti,t in enumerate(range(0,steps,fine_r)):
        print('t: ', t)
        for x in range(L):
            Cxt_[ti,x] = np.sum(np.sum((sp_a*np.roll(np.roll(sp_a,-x,axis=1),-ti,axis=0))[:-ti],axis=2))/(steps//fine_r+1 - ti)     #multiplying an array Sxt[t=ti] with a scalar Sxt[t=0,L=0]
    return Cxt_

#Cxt_2 = calc_cxt(sp_b, steps)

#Cxt_ = calc_cxt(sp_a, steps)
#X = np.arange(-L//2, L//2)
#print('T,N: ', T,N)
Sh = sp_a.shape; print(Sh)
Mq_1, Mq_2, Mq_3 = np.empty((Sh[0],Sh[1])), np.empty((Sh[0],Sh[1])), np.empty((Sh[0],Sh[1]))
#SMq_2 = np.empty(Cxt_2.shape)

T, N = Mq_1.shape

for ti in range(0,T):
    Mq_1[ti] = sff.fft(sp_a[ti,:,0])/N
    Mq_2[ti] = sff.fft(sp_a[ti,:,1])/N
    Mq_3[ti] = sff.fft(sp_a[ti,:,2])/N
    #SMq_2[ti] = 2*sff.fft(Cxt_2[ti,:])/N

Mq_ = np.sqrt(np.abs(Mq_1)**2 + np.abs(Mq_2)**2 + np.abs(Mq_3)**2) 
print('Mq_ shape: ', Mq_.shape)
Mq_00 = Mq_[0]
Mq_30 = Mq_3[0]
#print(Mq_00); print(Mq_30)

fig, (ax01, ax02) = plt.subplots(nrows=2,ncols=1)
ax01.plot(np.abs(Mq_00)); ax02.plot(np.abs(Mq_[100]));
ax01.set_xlim(0,8);
plt.savefig('./plots/trialplot.pdf')

print(np.abs(Mq_[0,0]))
print(np.abs(Mq_[50,0]))
print(np.where(np.abs(Mq_[:,0])/np.abs(Mq_[0,0]) < np.exp(-2)))

def plot_mkt():
	fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,figsize=(9,15))
	for k in range(1,3):
	    ax1.plot(np.arange(T), np.abs(Mq_3[:,k]), label=f'|M_3|; k = {k}') #fine_r * np.arange(T)
	    ax2.plot(np.arange(T), np.abs(Mq_[:,k]*Mq_00[k])/np.abs(Mq_00[0]*Mq_00[0]), label=f'MM corr; k = {k}') #fine_r * np.arange(T)
	    #ax1.plot(np.arange(T), np.abs(Mq_3[:,int(2**(k))]), label=f'M_3.M_3 corr; k = {int(2**k)}') #fine_r * np.arange(T)
	    #ax2.plot(np.arange(T), np.abs(Mq_[:,int(2**(k))]*Mq_00[int(2**(k))])/np.abs(Mq_00[0]*Mq_00[0]), label=f'MM corr; k = {int(2**k)}') #fine_r * np.arange(T)
	    #ax2.plot(np.arange(T), np.abs(SMq_2[:,int(2**(k-1))]), label=f'MM corr; inputq = 2; k = {int(2**k)}     
	    ax1.set_xlabel('t'); ax2.set_xlabel('t'); #ax1.set_ylim(0,0.6); ax2.set_ylim(0,0.6)
        #ax1.grid()
        #ax2.grid()
	plt.suptitle(r'$|M^{(3)}(k,t)|$ and $\frac{M(k,t) \cdot M(-k,0)}{|M(0,0)|^2} $')
	ax1.legend(); ax2.legend()
	plt.tight_layout()
	plt.savefig('./plots/SMqt_trial_modltdq{}_{}th.pdf'.format( m,j)) #len(filename['stagg'])))
	#plt.savefig('./plots/SMqt_vs_t_drvn_ordered_modltdq{}_{}th.pdf'.format( m,j)) #len(filename['stagg'])))

def plot_cxt():
    plt.figure(figsize=(10,10))
    plt.plot(X, Cxt)
    plt.xlabel('x')
    plt.ylabel(r'$C(x,t)$')
    #plt.title(r'$\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu)) #$D(x,t)$ heat map; $t$ v/s $x$;
    plt.savefig('./plots/Cxt_L{}_Lambda_{}_Mu_{}_dt_{}_ordered.png'.format(L,Lambda,Mu,dtstr, len(filename)))
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

plot_mkt()
"""
