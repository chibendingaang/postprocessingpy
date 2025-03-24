
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.stats import linregress


plt.style.use('matplotlibrc')
np.set_printoptions(threshold=2561)

L, dtsymb = map(int,sys.argv[1:3])
Lambda, Mu = map(float, sys.argv[3:5])
epss, choice = map(int, sys.argv[5:7])
#jumpval = int(sys.argv[7])

#hidden = int(sys.argv[7])

alpha = (Lambda - Mu)/(Lambda + Mu)
N_array = np.concatenate((np.array([0]), np.power(2, np.arange(0,7)), np.array([96, 112, 120, 124, 126, 127])))

# detect what the N is from the input .npy arrays, then decide whether to take the entire N_array
if alpha > 1:N_array = N_array[:8] # N_array = N_array[7:] 
elif alpha>0 and alpha<1:  N_array = N_array[7:] #N_array = N_array[:8] 
hiddensubfolder =  'alpha_ne_pm1/waveN' #if hidden in (-1)*N_array else 'alpha_ne_pm1'
#print(hiddensubfolder)

versions = dict()
versions.update(zip(N_array, ['v7'+str(nth) for nth in N_array]))

dt = 0.001 * dtsymb
dtstr = f'{dtsymb}emin3'
epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
# the lambda function was not defined for three digit or longer strings 
if alpha<1: alpha_deci = '975'
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr : ', alphastr)

if choice == 0:
    param = 'xpdrvn' if Lambda == 0 and Mu == 1 else 'xphsbg' if Lambda == 1 and Mu == 0 else 'xpa2b0'
    path = f'./xpdrvn/L{L}/{dtstr}'

elif choice == 1:
    param = 'qwdrvn' if Lambda == 0 and Mu == 1 else 'qwhsbg' if Lambda == 1 and Mu == 0 else 'qwa2b0'
    path = f'./{param}/{dtstr}'

elif choice == 2:
    param = 'drvn' if Lambda == 0 and Mu == 1 else 'hsbg' if Lambda == 1 and Mu == 0 else 'qwa2b0'
    path = f'./{param}/{dtstr}'

def auc_simpson(x,f):
    a = x[0]; b = x[-1]; n = len(x)
    h = (b-a)/(n-1)
    auc = (h/3)*(f[0]  + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])
    return auc


def opencorr(L,fine_res=5):
    Cxtpath1 = f'Cxt_series_storage/lL{L}/{hiddensubfolder}'
    #if hidden>0:
    #    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{fine_res}.txt'
    #else: filerepo = f'./{Cxtpath1}/outall_{alphastr}.txt'
    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{fine_res}_prox.txt'
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])#[10800:]
    f.close()
    f_last = filename[-1]

    Cxtk = np.load(f'./{Cxtpath1}/{f_last}', allow_pickle=True)
    #steps = int(Cxtk.size/(L+1))
    Cxtavg = np.zeros_like(Cxtk) #((steps,L+1))

    for k, fk  in enumerate(filename):
        Cxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)#[:steps]
        #print('Cxt, N: ', filename[k], Cxtk.shape[0])
        if not np.isnan(Cxtk).any():
            Cxtavg += Cxtk
            #if k%1000 ==0:
            #    print('prints the 1000the configuration: ', Cxtk)
        else:
            print('NaN value detected at file-index = ', k)
    return Cxtavg, filename

from scipy.ndimage import gaussian_filter1d
#step1: create the gaussian smoothened curve for the Correlator
#detect it's peak with np.max(Cxt_)


def plot_corrxt(ni,n):
    plt.figure()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    #ax2 = axes.inset_axes([0.125, 0.675, 0.25, 0.25])
    #ax3 = axes.inset_axes([0.675, 0.675, 0.25, 0.25])
    ax4 = axes[0].inset_axes([0.325, 0.125, 0.25, 0.25])
    ax5 = axes[1].inset_axes([0.325, 0.125, 0.25, 0.25])
    
    Cnnxt_, filenam2 = opencorr(L)
    N, steps, x_ = Cnnxt_.shape 
    print('Cxtavg.shape: ', N, steps, x_)
    print(steps, x_)
    
    countfiles = 20*filenam2.shape[0] 
    #averaging sample-size chosen =20 ; arbitrary, but stick with it!

    Cnnxtavg = Cnnxt_[ni]
    Cnewxt = Cnnxtavg / (countfiles)

    print('Cnnxtavg shape: ', Cnewxt.shape)
    T = Cnewxt.shape[0]

    safe_dist_r = L-n-1
    safe_dist_l = -n

    x_arr = np.arange(safe_dist_l, safe_dist_r +1)

    # when drift .neq. 0, the split is not x//2 : x//2; 
    #actually will be L -n: n sites on right:left
    corrxt_ = np.concatenate((Cnewxt[:,L - n:], Cnewxt[:, :L-n]), axis=1)
    print(corrxt_.shape)


    Tth_step = np.concatenate((np.arange(0,100,5), np.arange(100, 250, 10),np.arange(250,1250, 25),np.arange(1250,1400,10),  np.arange(1400, 1501, 5)))
    #Tth_step = np.arange(25,1601, 25, dtype=np.int32)
    Cxt_x0 = []
    Cxt_max_xeq = []
    
    for ti,t in enumerate(Tth_step): #np.arange(0,T):
        z = corrxt_[ti,:]
        if ti==0: print('C(x=0,t=0): ', corrxt_[0, n]) #as right boundary is closer

        print('time, x, y shapes: ', ti, x_arr.shape, z.shape)
        y = z/corrxt_[0,n] 

        Y_smoothened_G = gaussian_filter1d(y, sigma=2)
        Cxt_x0.append(np.max(z))     

        x0_loc = np.argmax(Y_smoothened_G[L//2:]) -L//2-1 #if Cx_.shape is L+1, use L//4 instead of L//2 (shape: 2*L-1)
        Cxt_max_xeq.append(x0_loc)
        #Cxt_max_xeq.append(np.max(x0_loc, -L//2 + np.abs(x0_loc)))	 #- L//2   

        '''df = pd.DataFrame({'x': x, 'y':y})
        window_size = 4
        df['y_smooth'] = df['y'].rolling(window=window_size).mean()'''
        print(f'C(0,t = {t})' , y[L//2])
        auc = auc_simpson(x_arr, y)
        print("area under the curve: ", auc)

        #axes00.plot(df['x'], df['y_smooth'], label=f'{t}', linewidth=2)
        if t in [50,100,150,200,250, 300, 350,400]:#,125,150,200,250]:
            axes[0].plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)
            ax4.plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)
        if t in [750,800,850,900,950,1000,1100, 1200, 1300, 1400]: #[1200,1210,1220,1230,1240, 1250]:
            axes[1].plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)
            ax5.plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)            
            #ax2.plot(x_arr *y[L//2], y/y[L//2], label=f'{int(0.1*dtsymb*t)}', linewidth=1)
        #ax2.plot(x / ti**(0.5), ti**(0.5) * y, label=f'{dtsymb*jumpval*ti}', linewidth=1) #it was 4*ti earlier because t_ array was defined peculiarly
        #use -loglog_slope instead of 0.5
    dX_ = dtsymb*2*np.arange(corrxt_.shape[0], dtype=np.int32)[:60]
    dY_ = corrxt_[:60, L//2]/corrxt_[0,L//2]
    slope, intercept, r_value, p_value, std_err = linregress(dX_, dY_)
    print(f"slope of d(corrxt(x=0))/dt: {slope:.4f}")
    #ax3.loglog(dtsymb*2*np.arange(corrxt_.shape[0], dtype=np.int32)[:25], corrxt_[:25, L//2]/corrxt_[0,L//2], '.k', linewidth=1.5)
    #ax3.scatter(x, y, label="Data", s=10)  # Scatter plot of data
    #ax3.plot(x, slope * x + intercept, color="red", label=f"Fit: y = {slope:.4f}x + {intercept:.4f}")    
    print("Cxt_max_xeq = ", Cxt_max_xeq)
    #ax4.plot(dtsymb*0.1*Tth_step, Cxt_max_xeq, ".--", linewidth=1)
    #ax4.set_title(rf'Peak of $C(x_0,t)$ v/s $t$')
    #ax5.plot(dtsymb*0.1*Tth_step, Cxt_max_xeq, ".--", linewidth=1)
    #ax5.set_title(rf'Peak of $C(x_0,t)$ v/s $t$')
    
    #ax4.plot(4*np.arange(corrxt_.shape[0])[1:-3], func_, '-.', linewidth=1.5)
    #ax4.plot(4*np.arange(corrxt_.shape[0])[1:-3], f1, '-.', linewidth=1.5)
    #axes00.set_ylim(-0.5, 1.25)
    #axes00.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='center left', ncol=2)
    #axes00.set_xlim(max(-32, -3*L//8), min(3*L//8, 32))
    #axes.set_ylim(-0.33, 1.05)
    ax4.set_xlim(safe_dist_l, min(L//8, L-n))
    ax5.set_xlim(safe_dist_l, min(L//8, L-n))
    #ax4.set_ylim(-0.25, 1.25)
    #ax5.set_ylim(-0.25, 1.25)
    location_legnd = 'upper right' if alpha>1 else 'upper left'
    axes[0].legend(title=r'$\mathit{t} $', fancybox=True, shadow=True, borderpad=1, loc=f'{location_legnd}', ncol=2)
    axes[1].legend(title=r'$\mathit{t} $', fancybox=True, shadow=True, borderpad=1, loc=f'{location_legnd}', ncol=2)
    #axes.set_xlim(max(-64, -6*L//8), min(6*L//8, 64))
    #ax2.set_xlim(max(-L//16,-33), min(33, L//16) )
    #ax2.yaxis.set_label_position("right")


    #if param == 'qwdrvn':
    #    axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    #axes00.set_xlabel(rf'$\mathbf{{x}} $')
    #axes00.set_ylabel(rf'$\mathbf{{C(x,t)}}$')
    axes[0].set_title(rf'$n={n}$')
    axes[0].set_xlabel(rf'$\mathbf{{x}} $')
    axes[0].set_ylabel(rf'$\mathbf{{C_n(x,t)}}$')
    axes[1].set_xlabel(rf'$\mathbf{{x}} $')
    axes[1].set_ylabel(rf'$\mathbf{{C_n(x,t)}}$')    
    fig.tight_layout()

    plt.savefig('./plots/CxtN_vs_x_L{}_{}_{}_{}_prox_{}configs_{}.pdf'.format( L, param, epstr[epss], alphastr, countfiles, versions[n]))

#opencorr(L)
for ni, n in enumerate(N_array):
    plot_corrxt(ni, n)

