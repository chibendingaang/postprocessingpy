
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

hidden = int(sys.argv[7])

hidden_subfolders = dict()
hidden_drifts = [0, -1, -2, -3, -4, -8, -32, -64, -96, -112,-120]
hidden_subfolders.update(zip(hidden_drifts, zip(['alpha_ne_pm1/drift' + str(np.abs(i)) for i in hidden_drifts], ['v5' + str(np.abs(i)) for i in hidden_drifts])))

hidden_subfolders.update({1: ('alpha_ne_pm1/alternate_PDas', 'v32')})
hidden_subfolders.update({10: ('alpha_ne_pm1', 'v5')})
hidden_subfolders.update({25: ('alpha_ne_pm1', 'v5')})
"""if hidden ==-3:
    version = 'v54';
    hiddensubfolder = 'alpha_ne_pm1/drift3_PDas'
elif hidden ==-2:
    version = 'v53';
    hiddensubfolder = 'alpha_ne_pm1/drift2_PDas'
elif hidden ==-1:
    version = 'v52';
    hiddensubfolder = 'alpha_ne_pm1/drift_PDas' #v42; reversed_PDas
elif hidden==1:
    version = 'v32'  
    hiddensubfolder = 'alpha_ne_pm1/alternate_PDas'
else: 
    version = 'v22'
    hiddensubfolder = 'alpha_ne_pm1'
"""
hiddensubfolder, version = hidden_subfolders[hidden]
print('hiddensubfolder, version: ', hiddensubfolder, version)

dt = 0.001 * dtsymb
dtstr = f'{dtsymb}emin3'
epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

alpha = (Lambda - Mu)/(Lambda + Mu)
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

    if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt'
    if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt'

    #if hidden>0:
    #    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{fine_res}.txt'
    #else: filerepo = f'./{Cxtpath1}/outall_{alphastr}.txt'
    filerepo = f'./{Cxtpath1}/outall_{alphastr}_jump{fine_res}.txt'
    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])#[18800:]
    f.close()
    f_last = filename[-1]

    Cxtk = np.load(f'./{Cxtpath1}/{f_last}', allow_pickle=True)
    #steps = int(Cxtk.size/(L+1))
    Cxtavg = np.zeros_like(Cxtk) #((steps,L+1))

    for k, fk  in enumerate(filename):
        Cxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)#[:steps]

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


def plot_corrxt(magg):
    plt.figure()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    #ax2 = axes.inset_axes([0.125, 0.675, 0.25, 0.25])
    #ax3 = axes.inset_axes([0.675, 0.675, 0.25, 0.25])
    ax4 = axes[0].inset_axes([0.325, 0.125, 0.25, 0.25])
    ax5 = axes[1].inset_axes([0.325, 0.125, 0.25, 0.25])
    
    Cnnxt_, filenam2 = opencorr(L)
    steps, x_ = Cnnxt_.shape 
    print('Cxtavg.shape: ')
    print(steps, x_)
    
    countfiles = filenam2.shape[0]

    Cnnxtavg = Cnnxt_
    Cnnxtavg = Cnnxtavg / (countfiles)
    Cnnnewxt = Cnnxtavg

    print('Cnnxtavg shape: ', Cnnxtavg.shape)
    T = Cnnxtavg.shape[0]
    #t = t0 - t0 % 16
    #t_ = np.array([1,2,3,4,5,6])

    x_arr = np.arange(- np.abs(hidden), x_ - np.abs(hidden))

    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization', 'ab': 'Hybrid'}

    corr = {'stagg': 'Cnnxt', 'ab': 'Cmhyb_xt', 'magg': "Cxt" }
    #for any hiddendrift .neq. 0, the split is not x//2 : x//2; 
    #actually will be L -abs(hidden): abs(hidden) sites on right:left
    corrxt_ = np.concatenate((Cnnnewxt[:,x_ - np.abs(hidden):], Cnnnewxt[:, :x_ - np.abs(hidden)]), axis=1)
    print(corrxt_.shape)


    Tth_step = np.concatenate((np.arange(0,200,5), np.arange(200,400,10), np.arange(400, 1200, 25), np.arange(1200,1400,10), np.arange(1400,1501,5)))
    #Tth_step = np.arange(25,1601, 25, dtype=np.int32)
    Cxt_x0 = []
    Cxt_max_xeq = []
    
    for ti,t in enumerate(Tth_step): #np.arange(0,T):
        z = corrxt_[ti,:]
        if ti==0: print('C(x=0,t=0): ', corrxt_[0, np.abs(hidden)-1]) #as right boundary is closer
        #L-1-np.abs(hidden) when left boundary is closer
        print('time, x, y shapes: ', ti, x_arr.shape, z.shape)
        y = z/corrxt_[0,np.abs(hidden)-1] 
        #the value at delta_x = 0, basically 
        #L//2 if Cxt_.shape = L+1 ; L if Cxt_.shape = 2*L-1
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
        if t in [50,60,70,80,90,100]:#,125,150,200,250]:
            axes[0].plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)
            ax4.plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)
        if t in [1400,1420,1440,1460,1480, 1500]: #[1200,1210,1220,1230,1240, 1250]:
            axes[1].plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)
            ax5.plot(x_arr, Y_smoothened_G, label=f'{int(dtsymb*0.1*t)}', linewidth=1.2)            
            #ax2.plot(x_arr *y[L//2], y/y[L//2], label=f'{int(0.1*dtsymb*t)}', linewidth=1)
        #ax2.plot(x / ti**(0.5), ti**(0.5) * y, label=f'{dtsymb*jumpval*ti}', linewidth=1) #it was 4*ti earlier because t_ array was defined peculiarly
        #use -loglog_slope instead of 0.5
    dX_ = dtsymb*2*np.arange(corrxt_.shape[0], dtype=np.int32)[:60] 
    """use Tth_step instead of corrxt_.shape[0] since the former is not uniformly distributed array
    """
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
    ax4.set_xlim(-L//8, min(L//8, L-np.abs(hidden)))
    ax5.set_xlim(-L//8, min(L//8,L-np.abs(hidden)))
    #ax4.set_ylim(-0.25, 1.25)
    #ax5.set_ylim(-0.25, 1.25)
    axes[0].legend(title=r'$\mathit{t} $', fancybox=True, shadow=True, borderpad=1, loc='upper left', ncol=2)
    axes[1].legend(title=r'$\mathit{t} $', fancybox=True, shadow=True, borderpad=1, loc='upper left', ncol=2)
    #axes.set_xlim(max(-64, -6*L//8), min(6*L//8, 64))
    #ax2.set_xlim(max(-L//16,-33), min(33, L//16) )
    #ax2.yaxis.set_label_position("right")


    #if param == 'qwdrvn':
    #    axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    #axes00.set_xlabel(rf'$\mathbf{{x}} $')
    #axes00.set_ylabel(rf'$\mathbf{{C(x,t)}}$')
    axes[0].set_title(rf'$n={np.abs(hidden)}$')
    axes[0].set_xlabel(rf'$\mathbf{{x}} $')
    axes[0].set_ylabel(rf'$\mathbf{{C(x,t)}}$')
    axes[1].set_xlabel(rf'$\mathbf{{x}} $')
    axes[1].set_ylabel(rf'$\mathbf{{C(x,t)}}$')    
    fig.tight_layout()

    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_{}_{}configs_{}.pdf'.format(corr[magg], L, param, epstr[epss], alphastr, countfiles, version))

plot_corrxt('ab')

