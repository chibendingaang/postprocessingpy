#from scipy.fft import fft, ifft, rfft, irfft, fftfreq
import scipy.fftpack as sff
import numpy as np
import matplotlib.pyplot as plt
import sys

#the input sinusoidal modulation is in the Z-X plane

L = 512
dt = 0.002
param = 'xpdrvn'
dtstr = '2emin3'
begin = int(sys.argv[1])

#x = np.arange(1, 50, 0.25)
#y = 2*np.sin(x) + 2*np.cos(2*x) - np.cos(x/2)

path = f'./{param}/L{L}/{dtstr}'
Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(begin)))
steps = int((Sp_aj.size/(3*L)))
print('steps = ', steps)

Sp_a = np.reshape(Sp_aj, (steps, L, 3))
Y = Sp_a
X = np.arange(L)

N = Y.shape[1]
T = Y.shape[0]
x = np.arange(L)
Xf = sff.fftfreq(N) #uniform sampling is due to the uniform lattice spacing
Sp1_q = np.zeros(Y.shape[:2])
Sp2_q = np.zeros(Y.shape[:2])
Sp3_q = np.zeros(Y.shape[:2])
Strq_fact = np.zeros((T,4))

for ti in range(0,T):
    Sp1_q[ti] = 2*sff.fft(Y[ti,:,0])/N
    Sp2_q[ti] = 2*sff.fft(Y[ti,:,1])/N
    Sp3_q[ti] = 2*sff.fft(Y[ti,:,2])/N
#print('Sp_qn_-qn[ti=100]: ', Sp2_q[100,4], Sp2_q[100,-4])


for qi,q in enumerate([1,2,4,8]):
    for ti in range(T):
        Strq_fact[ti,qi] = np.sum((Sp1_q[:,q]*np.roll(Sp1_q[:,-q],-ti,axis=0))[:-ti] + (Sp2_q[:,q]*np.roll(Sp2_q[:,-q],-ti,axis=0))[:-ti] + (Sp3_q[:,q]*np.roll(Sp3_q[:,-q],-ti,axis=0))[:-ti])/(T-ti)
print(Strq_fact.shape)
print(Strq_fact)

#Spin_f = sff.fft(Y)
#S1f = Spin_f[:,:,0] 
#S2f = Spin_f[:,:,1] 
#S3f = Spin_f[:,:,2]


mode = begin//1000

fig,axes = plt.subplots(nrows=3, ncols=2, figsize=(16,20))
for ti in [0]: #range(0) : #,T,800):
    axes[0,0].plot(X, Sp_a[ti,:,0])
    axes[0,1].plot(Xf, np.abs(Sp1_q[ti,:])); axes[0,1].set_ylabel('S1(q,t)'); axes[0,1].set_xlim(-0.05,0.05)

    axes[1,0].plot(X, Sp_a[ti,:,1])
    axes[1,1].plot(Xf, np.abs(Sp2_q[ti,:])); axes[1,1].set_ylabel('S2(q,t)'); axes[1,1].set_xlim(-0.05,0.05)


    axes[2,0].plot(X, Sp_a[ti,:,2])
    axes[2,1].plot(Xf, np.abs(Sp3_q[ti,:])); axes[2,1].set_ylabel('S3(q,t)'); axes[2,1].set_xlim(-0.05,0.05)

plt.tight_layout()
plt.savefig('./plots/Sqpeaks_qinit_{}{}_xpdrvn.png'.format(mode,mode))

plt.figure()
for k in range(4):
    plt.plot(np.arange(T), Strq_fact[:,k], label=f'k={k}-mode')
plt.legend()
plt.title(r'$<\vec{S}(k,0) \vec{S}(-k,t)>$')
plt.savefig('./plots/Strqt_qinit_{}{}_xpdrvn.png'.format(mode, mode))

"""
fig,(ax1,ax3) = plt.subplots(2, figsize=(10, 15))
for t in range(0, steps):
    y = Y[t,:]
    yf = fft(y)
    print(yf.shape)
    xf = fftfreq(N, 1/0.25)#[:N//2]
    #mode = begin//1000
    #fy = irfft(fx)
    #ax1.plot(x, y, label=t)
    #ax1.plot(x, len(x)*[0], 'k')
    #ax2.plot(yf, label=t)
    #ax3.plot(xf, 2.0/N * np.abs(yf[0:N//2]), label=t)


ax1.set_title('y vs x')
ax1.legend()
#ax2.set_title('F.T. (y(x))')
#ax2.legend()
ax3.set_title('F.T. frequencies')
ax3.legend()
plt.savefig('./plots/fig_freqx_qinit{}{}_xpdrvn.png'.format(mode,mode))
"""
