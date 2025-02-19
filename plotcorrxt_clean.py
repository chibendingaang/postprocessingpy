
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.interpolate import CubicSpline
import pandas as pd

plt.style.use('matplotlibrc')
np.set_printoptions(threshold=2561)

L, dtsymb = map(int,sys.argv[1:3])
Lambda, Mu = map(float, sys.argv[3:5])
epss, choice = map(int, sys.argv[5:7])
#jumpval = int(sys.argv[7])

hidden = int(sys.argv[7])
hiddensubfolder = 'alpha_ne_pm1/alternate_PDas' if hidden == 1 else 'alpha_ne_pm1'
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

def opencorr(L):
    Cxtpath1 = f'Cxt_series_storage/lL{L}/{hiddensubfolder}'

    if L ==2048 and dtsymb == 1: filerepo = f'./{Cxtpath1}_yonder/outall.txt'
    if L == 2048 and dtsymb == 2: filerepo = f'./{Cxtpath1}_yonder/poutall.txt'

    filerepo = f'./{Cxtpath1}/outall_{alphastr}.txt'

    f = open(filerepo)
    filename = np.array([name for name in f.read().splitlines()])[:99960]
    f.close()
    f_last = filename[-1]

    Cxtk = np.load(f'./{Cxtpath1}/{f_last}', allow_pickle=True)
    #steps = int(Cxtk.size/(L+1))
    Cxtavg = np.zeros_like(Cxtk) #((steps,L+1))

    for k, fk  in enumerate(filename):
        Cxtk = np.load(f'./{Cxtpath1}/{fk}', allow_pickle=True)#[:steps]

        if not np.isnan(Cxtk).any():
            Cxtavg += Cxtk
            if k%1000 ==0:
                print('prints the 1000the configuration: ', Cxtk)
        else:
            print('NaN value detected at file-index = ', k)

    return Cxtavg, filename

def plot_corrxt(magg):
    plt.figure()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 9))
    #ax2 = axes.inset_axes([0.125, 0.675, 0.25, 0.25])
    #ax3 = axes.inset_axes([0.675, 0.675, 0.25, 0.25])

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

    x = np.arange(-x_//2, x_//2)
    X = np.arange(x_//2)

    magstr = {'magg': 'Magnetization', 'stagg': 'Staggered magnetization', 'ab': 'Hybrid'}

    corr = {'stagg': 'Cnnxt', 'ab': 'Cmhyb_xt', 'magg': "Cxt" }

    corrxt_ = np.concatenate((Cnnnewxt[:T,x_//2:], Cnnnewxt[:T, :x_//2]), axis=1)
    print(corrxt_.shape)

    for ti in np.arange(0,T):
        z = corrxt_[ti,:]
        print('time, x, y shapes: ', ti, x.shape, z.shape)
        y = z/corrxt_[0,L//2]
        df = pd.DataFrame({'x': x, 'y':y})

        window_size = 1

        df['y_smooth'] = df['y'].rolling(window=window_size).mean()
        print(f'C(0,t = {10 + 10*ti})' , y[L//2])
        auc = auc_simpson(x, y)
        print("area under the curve: ", auc)

        axes.plot(df['x'], df['y_smooth'], label=f'{10+10*ti}', linewidth=2)

    axes.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='center left', ncol=2)
    axes.set_xlim(max(-32, -3*L//8), min(3*L//8, 32))
    #ax2.set_xlim(max(-L//16,-33), min(33, L//16) )
    #ax2.yaxis.set_label_position("right")

    #if param == 'qwdrvn':
    #    axes.set_xlim(-75, 75); ax2.set_xlim(-10,10)
    axes.set_xlabel(rf'$\mathbf{{x}} $')
    axes.set_ylabel(rf'$\mathbf{{C(x,t)}}$')

    fig.tight_layout()
    version = 'v32' if hidden==1 else 'v22'
    plt.savefig('./plots/{}_vs_x_L{}_{}_{}_{}_{}configs_{}.pdf'.format(corr[magg], L, param, epstr[epss], alphastr, countfiles, version))

plot_corrxt('ab')

