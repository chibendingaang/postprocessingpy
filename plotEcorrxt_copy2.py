import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')
import pandas as pd

def opencorr(L, param):
    path = f'CExt_series_storage/L{L}'
    filerepo = f'./{path}/outcxt{param}_545K.txt' #outcxtqwhsbg_545K.txt'
    f = open(filerepo)
    filename = np.array([x for x in f.read().splitlines()])
    f.close()

    steps = min(int(np.load(f'./{path}/{filename[-1]}').size / L), 1280)
    Cxtavg = np.zeros((steps, L))

    for fk in filename:
        Cxtk = np.load(f'./{path}/{fk}', allow_pickle=True)[:steps]
        Cxtavg += Cxtk
    print('Cxtk first, second entries: ', Cxtk[0], Cxtk[1])
    #the first entry Cxtk[0] is all zeros: 
    #note this oddity - we basically saved the data from the second step onwards in calc_corrxt file
    #forgetting about the index numbering for python arrays
    return filename, Cxtavg / len(filename)

def plot_corrxt(Cxtavg):
    x_ = Cxtavg.shape[1]
    x = np.arange(x_) - int(0.5 * x_)

    t = Cxtavg.shape[0]
    #t_ = np.concatenate((np.arange(t // 16, t // 4, t // 16),
    #                     np.arange(3 * t // 8, t // 2 + 1, t // 8),
    #                     np.arange(6 * t // 8, 7 * t // 8 + 1, t // 4)))
    t_ = np.array([8, 10, 16, 20, 32, 40]) #([16,24, 32, 48, 64, 96]) #t//16, 2*t//16, 3*t//16, 4*t//16, 6*t//16, 8*t//16, 12*t//16])
    Cnewxt = np.concatenate((Cxtavg[:, L // 2:], Cxtavg[:, 0:L // 2]), axis=1)

    fig, ax1 = plt.subplots(figsize=(11, 9))
    ax2 = ax1.inset_axes([0.125, 0.675, 0.25, 0.25])
    ax3 = ax1.inset_axes([0.675, 0.675, 0.25, 0.25])
    print('t_ : ' , t_)
    for ti in t_:
        y = Cnewxt[ti]
        z = y/Cnewxt[1, L//2]
        df = pd.DataFrame({'x': x, 'z':z})
        window_size = 4
        df['z_smooth'] = df['z'].rolling(window=window_size).mean()
        ax1.plot(df['x'], df['z_smooth'], label=f'{4 * ti}', linewidth=2.5)
        ax2.plot(x / ti ** (0.5), ti ** (0.5) * df['z_smooth'], label=f'{4 * ti}', linewidth=2)
    ax3.loglog(4*np.arange(Cnewxt.shape[0])[1:-3], Cnewxt[1:-3, L // 2]/Cnewxt[1,L//2], '.k', linewidth=2)
    dy_ = np.log(Cnewxt[-10, L//2]/Cnewxt[10, L//2])
    dx_ = np.log(np.arange(t)[-10]/np.arange(t)[10])
    print('slope of dcorrxt/dt: ', dy_/dx_)
    
    tp = t_[0] 
    z2max = (tp)**0.5*Cnewxt[tp,L//2]/Cnewxt[1,L//2]
    #ax1.legend(title=r'$t = $')
    ax1.legend(title=r'$\mathit{t} = $', fancybox=True, shadow=True, borderpad=1, loc='center right', ncol=2)
    if param == 'qwhsbg': ax1.set_xlim(-200,200)
    if param == 'qwdrvn': ax1.set_xlim(-75,75)
    ax1.set_xlabel(r'$\mathbf{x}$')
    ax1.set_ylabel(r'$\mathbf{C(x,t)}$')

    if param == 'qwhsbg': ax2.set_xlim(-25,25)
    if param == 'qwdrvn': ax2.set_xlim(-10,10)
    ax2.set_ylim(-0.05,z2max + 0.05)
    ax2.set_xlabel(r'$\mathbf{x/t^{0.5}}$')
    ax2.set_ylabel(r'$\mathbf{t^{0.5}C(x,t)}$')
    ax3.set_xlabel(r'$\mathbf{t}$')
    ax3.set_ylabel(r'$\mathbf{C(0,t)}$')
    if param == 'qwhsbg': CE = 'CExt'
    if param == 'qwdrvn': CE = 'CENxt'
    fig.tight_layout()
    plt.savefig(f'./plots/{CE}_vs_x_L{L}_{param}_{len(filename)}configs_v8.pdf')
    #plt.show()

if __name__ == "__main__":
    L = int(sys.argv[1])
    lamda, mu = map(int, sys.argv[2:4])
    if lamda == 1 and mu ==0: param = 'qwhsbg'
    if lamda == 0 and mu == 1: param= 'qwdrvn'
    filename, Cxtavg = opencorr(L, param)
    plot_corrxt(Cxtavg)

