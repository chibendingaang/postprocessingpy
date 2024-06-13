#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:46:47 2021

@author: nisarg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import time 
plt.style.use('matplotlibrc')

start = time.perf_counter()
L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = dtsymb*10**(-3)
dtstr = f'{dtsymb}emin3'
Lambda, Mu, begin, end  = map(int,sys.argv[3:7])
#interval = 1
t_fact = int(10/dtsymb)


#kksmall = np.arange(begin,end,interval)
#kk = np.concatenate((np.arange(351,401),np.arange(begin,end)))

if Lambda == 1 and Mu == 0: 
    param = 'qwhsbg' #'hsbg'
    if dtsymb == 2: filesize = '81M'
    if dtsymb == 1: filesize = '201M'; 
    steps = 5*L//8
    Axlim2 = 2.4
    ax2lim2 = 200
if Lambda == 0 and Mu == 1: 
    param = 'qwdrvn' #'drvn'
    if dtsymb == 2: filesize = '101M'
    if dtsymb == 1: filesize = '161M'; begin = begin + 1297; end = end + 1297
    steps = 25*L//32
    Axlim2 = 2.0
    ax2lim2 = 160

#filerepo = 'splitdecorr{}.txt'.format(param); 
filerepo = f'./Dxt_storage/emin3_{param}/out2048_{dtstr}.txt'
f = open(filerepo)
print('filerepo name: ', filerepo)
kksmall = np.array([x for x in f.read().splitlines()])[begin:end] #([x for x in f.readlines()]) #filename
f.close()
print('shape of kksmall : ', kksmall.shape)
print('first and last elements in kksmall : ', kksmall[0], kksmall[-1])

epss = 3 #int(sys.argv[6])
epsilon = 10**(-1.0*epss) #0.00001

filesteps = np.load(f'./Dxt_storage/emin3_{param}/{filesize}_{dtstr}/{kksmall[-1]}', allow_pickle=True).shape[0]
steps = min(filesteps, steps)
Dxtavg = np.zeros((2*steps+1,L)) #2*steps+1

for k in kksmall :
	Dxtk = np.load(f'./Dxt_storage/emin3_{param}/{filesize}_{dtstr}/{k}', allow_pickle=True);
	Dxtavg += Dxtk[:2*steps+1]
#Dxtk = np.load('Dxt_{}_lamda_{}_mu_{}_sample_11001to11060.npy'.format(L, Lambda,Mu))
print('# of files for averaging : ', len(kksmall))
Dxtavg = Dxtavg/len(kksmall)
D_th = 100*epsilon**2
#Dxtavg is the concatenated symmetric decorrxt plot
Dnewxt = Dxtavg[:,L//2:] #obtainspins(steps*int(1./dt))
print('Dnewxt : ', Dnewxt)
logDxt = np.log(Dnewxt/epsilon**2)
print('logDxt shape: ' , logDxt.shape)
#vel_inst = np.empty(steps*int(1./dt))
x = np.arange(0, L//2)

#plt.figure()
#fig, axes = plt.subplots(figsize=(8,7))
#for t in range(100,850,100):
#    axes.plot(x, Dnewxt[t], label=f'{t//5}', linewidth=1.5)
#fig.savefig('./plots/newlogDxt_{}_{}configs.pdf'.format(param, end-begin))


plt.figure()
fig, axes = plt.subplots(figsize=(8,7))
ax2 = axes.inset_axes([0.125, 0.125, 0.525, 0.325])
ax2.tick_params(axis='both', which='both', direction='in', width=1.2)

"""This procedure doesn't work: 
the one where we take slop of func_empir near x = VB.t
kappa_Lyap = 0
for t in range(20,2560, 10):
    if param == 'qwdrvn': x1 = int(1.2*t//5); x2 = int(1.4*t//5); VB = 1.35
    if param == 'qwhsbg': x1 = int(1.5*t//5); x2 = int(1.8*t//5); VB = 1.64
    DX = (5/(VB*t))**2*(x1+x2)*(x1-x2) # (5*x[x1]/(VB*t))**2) -  (5*x[x2]/(VB*t))**2
    DY = 2.5*(logDxt[t-1,x2] - logDxt[t-1,x1])/t
    kappa_Lyap += DY/DX
    print('DY, DX : ', DY, DX)
kappa_Lyap = kappa_Lyap/len(range(20,2560,10))

"""

def f_empirical(x,t):
    return kappa_Lyap*(1 - (t_fact*x/(VB*t))**2)

#use the following function to guesstimate V_butterfly better
def check_x_intercept(func):
    x_intercepts = []
    for i, fi in enumerate(func[0:-1]):
        if fi*func[i+1] < 0: x_intercepts.append(i)
    return x_intercepts[0]

#t = 20 #either 11 or 20
#if param == 'qwdrvn': x1 = int(1.2*t//5); x2 = int(1.4*t//5); VB = 1.35
#if param == 'qwhsbg': x1 = int(1.5*t//5); x2 = int(1.8*t//5); VB = 1.64

VB = 0
for t in range(201,steps):
    v = t_fact*check_x_intercept(0.5*t_fact*logDxt[t]/t)/t
    # above is just v = dx/dt; dx is obtained by calling check_x_intercept func
    VB += v
    #print('t, logdxtbyepssq : ', t, 2.5*logDxt[t-1,0]/t)
VB = VB/len(range(201,steps))
print(' v_i = ', v)
print('vB = ', VB)

#print('t, logDxt : ', 1, 2.5*logDxt[0]/t)
tinit = int(sys.argv[7]) #11-25; 20 is most likely the origin point of the light-cone
kappa_Lyap = 0.5*t_fact*logDxt[tinit-1, 0]/tinit
print('t_init, Lyapunov exponent kappa : ', tinit, kappa_Lyap)

t_array = np.array([100, 150, 200,300, 400, 500]) 

handle1 = []
for ti,t  in enumerate(int(2/dtsymb)*t_array):
    ti, = axes.plot(t_fact*x/t, 0.5*t_fact*logDxt[t-1]/t, label=f'{int(t/t_fact)}', linewidth=1.5)
    ax2.plot(x, Dnewxt[t], linewidth=1)
    handle1.append(ti)
    #DY = 2.5*(logDxt[t-1,x2] - logDxt[t-1,x1])/t
    #DX = (5/(VB*t))**2*(x1+x2)*(x1-x2) # (5*x[x1]/(VB*t))**2) -  (5*x[x2]/(VB*t))**2
    #kappa_Lyap += DY/DX
    #print('DY, DX : ', DY, DX)


"""more reliable procedure:
extrapolate the value of func_empir at x = 0 for t=1, t=0 eventually
"""
#pick any legal value of t, this should work
t = 100
func_empir = f_empirical(x,t)

# Create a legend for the first line.
first_legend = axes.legend(handles=handle1, title=r'$t = $', ncol=2, loc='upper right')
#ax1 = axes.twinx()

empir_fit, = axes.plot(t_fact*x/t, func_empir, '--k', label=rf'${{{kappa_Lyap:0,.2f}}} (1 - (v/{{{VB:0,.2f}}})^2)$',  linewidth=1.5)

# Add the legend manually to the Axes.
axes.add_artist(first_legend)
# Create another legend for the second line.
axes.legend(handles=[empir_fit], loc='upper center', frameon=False)

#ax1.get_yaxis().set_visible(False)
#axes.legend(loc = 'upper center')
#axes.handles=[line2], loc='lower right')

ax2.set_xlabel(r'$\mathit{x} $')
ax2.set_ylabel(r'$\mathit{D(x,t)} $')
axes.set_ylim(-0.5, 0.6)
ax2.set_xlim(0,ax2lim2)
ax2.set_ylim(-0.1, 1.1)
axes.set_xlabel(r'$\mathbf{x/t} $')
axes.set_ylabel(r'$\mathbf{\left[ln(D(x,t)/\varepsilon^2)\right]/(2t)} $')
axes.set_xlim(0,Axlim2)

#axes.set_ylim(-0.1, 1.1)
fig.savefig('./plots/newlogDxtcomplete_{}_{}configs_{}tinit.pdf'.format(param, len(kksmall), tinit))

"""
for t in np.arange(0,steps+1, 10):
    l = int(4*t)
    ti = t
    logDxt = np.zeros(l)
    counter = 0
    
    for x in range(10,l,10):
        logDxt = np.log(Dxtavg[int(ti),L//2:L//2+l]/epsilon**2)
    f = open('./logDxt/logDxtbyepssq_L{}_t{}_lambda_{}_mu_{}_eps_1emin3_{}config.npy'.format(L,t,Lambda, Mu, interval*len(kksmall)), 'wb')
    np.save(f, logDxt)
    f.close()
    #print ('t,       log(Dxt/eps*eps)/l' )
    #print()
    #print (t, logDxt/l)
"""

'''
plt.figure(figsize= (8, 7))
#fig.colorbar(img, orientation = 'horizontal')

#plt.pcolormesh(Dxtavg[:], cmap = 'seismic',vmin =  0, vmax = 1.0);
#plt.xlabel('x')
#plt.ylabel(r'$t$')
#plt.title(r'$\lambda = $ {}, $\mu = $ {}'.format(Lambda,Mu)) #$D(x,t)$ heat map; $t$ v/s $x$;
#plt.colorbar()
#plt.savefig('./checkdxtplots/Dxt_L{}_Lambda_{}_Mu_{}_eps_1emin3_dt_{}_{}confg.png'.format(L,Lambda,Mu,dtstr, interval*len(kksmall)))
#plt.show()
'''

print('time elapsed : ', time.perf_counter() - start)
