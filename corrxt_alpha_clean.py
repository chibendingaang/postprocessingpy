
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import os

np.set_printoptions(threshold=2561)

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 *dtsymb

Lambda, Mu = map(float,sys.argv[3:5])
begin, end = map(int, sys.argv[5:7])

epss = int(sys.argv[7])
epsilon = 10**(-1.0*epss)
choice = int(sys.argv[8])
#t_smoothness = int(sys.argv[9])
#hidden = int(sys.argv[9])

#hidden_subfolders = dict()
#hidden_drifts = [0, -1, -2, -3, -4, -8, -32, -64, -96, -120]
'''write a conditional that the arg #hidden has to be in hidden_drifts else break the program 
or better, don\'t invoke the hidden_subfolder dict'''
#hidden_subfolders.update(zip(hidden_drifts, ['alpha_ne_pm1/drift' + str(np.abs(i)) for i in hidden_drifts]))

hiddensubfolder = 'alpha_ne_pm1' #hidden_subfolders[hidden] if hidden in hidden_drifts else 

#if dtsymb ==2:
#    fine_res = 1*t_smoothness
#if dtsymb ==1:
#     fine_res = 2*t_smoothness

dtstr = f'{dtsymb}emin3'
epstr = {3: 'eps_min3', 4: 'eps_min4', 6: 'eps_min6', 8: 'eps_min8' }

alpha = (Lambda - Mu)/(Lambda + Mu)

alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
# the lambda function was not defined for three digit or longer strings 
if alpha<1: alpha_deci = '975'
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr)

path_to_directree = f'/home/nisarg/entropy_production/mpi_dynamik/xpa2b0/L{L}/alpha_{alphastr}/{dtstr}'

epstr = {3:'emin3', 4:'emin4', 6:'emin6', 8:'emin8', 33:'min3', 44:'min4'}

def label_param():
    if Lambda == 1 and Mu == 0: paramstm = 'hsbg'
    elif Lambda == 0 and Mu == 1: paramstm = 'drvn'
    else: paramstm = 'a2b0'

    if choice == 0:
        param = 'xp' + paramstm

        if paramstm!= 'a2b0':
            path =  f'./{param}/L{L}/2emin3'
        else:
            path =  f'./{param}/L{L}/alpha_{alphastr}/2emin3'
        path = path_to_directree

    if choice == 1:
        param = 'qw' + paramstm

        if paramstm!= 'a2b0':
            path =   f'./{param}/L{L}/2emin3'
        else:
            param = 'qwa2b0'; path =  f'./{param}/2emin3/alpha_{alphastr}'

    if choice ==2:
        if paramstm == 'a2b0':
            param = 'qw' + paramstm
            path =  f'./{param}/2emin3/alpha_{alphastr}'
        else:
            param = paramstm
            path =  f'./{param}/L{L}/2emin3'
    print('param: ', param)

    return param

param = label_param()

start  = time.perf_counter()


def calc_mconsv_ser(spin, alpha):
    alpha_ser = alpha**(-np.arange(L))[:, np.newaxis]  # broadcast the array to the correct shape (2-D: (L,1))
    return spin * alpha_ser
    #mconsv[:] = Sp_a[:]*alpha_ser
    #mdecay[ti,x] = 0.5*(Sp_a[ti,2*x] - Sp_a[ti,(2*x+1)%L]/alpha)/(alpha)**x

fine_res = 25
def calc_Cxt_optimized(steps, spin, alpha):
    #spin = spin[0:steps:fine_res]
    alpha_ = alpha
    T_array = np.concatenate((np.arange(0,100,5), np.arange(100, 250, 10),np.arange(250,1250, 25),np.arange(1250,1400,10),  np.arange(1400, 1501, 5)))
    T = T_array.shape[0]
    #T_array = np.array([20,40,60,80,100,120,160,200,240, 320]) #dtype=np.int32) #exclude delta_t = 0
    T = T_array.shape[0]
    
    Cxt_ = np.zeros((T, L+1)) #indices and timestep value are in 1:10 ratio
    CExt_ = np.zeros((T, L))

    print('Cxt.shape: ', Cxt_.shape)
    print('CExt.shape: ', CExt_.shape)

    """Spnbr_a = np.roll(Sp_a,-1,axis=1)

    E_loc = (-1)*np.sum(Sp_a*Spnbr_a,axis=2)
    energ_1 = np.sum(E_loc, axis = 1)
    eta_ = np.array([1,1,1,-1,-1,-1]*int(Sp_a.size/6))
    eta = np.reshape(eta_, Sp_a.shape)

    Sp_ngva = Sp_a*eta
    Spnbr_ngva = Spnbr_a*eta
    Eeta_loc = (-1)*np.sum(Sp_a*Spnbr_ngva,axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis = 1)
    """    
    
    #safe_dist = int(max(L//2, L*np.power(alpha,-L//2))) if alpha>1 else int(min(L//2, L*np.power(alpha,-L//2)))
    safe_dist = L//2
    r = int(1./dtsymb)
    t_l = fine_res
    #no point in taking less samples when the processing time advantage is negligible 
    #if hidden ==-3 else T_array[-1]
    #T_windows = int(steps/t_l)
    #j = file_j
    
    for ti,t in enumerate(T_array): #enumerate(range(0, steps, fine_res)):
        """
        depending on our choice, we can keep # of T_windows fixed to some value
        or as in the following case, fewer windows T-separation increases
        """
        #if hidden==-3:
        spin_t = spin[0:-t:t_l] if t>0 else spin[::t_l] #t < T_array[-1] else spin[:-t_l:t_l]
        spin_t_shifted = spin[t::t_l]
        #else: #no point in taking less samples when the processing time advantage is negligible 
        #    spin_t = spin[0:-t:t] # if ti>0 else spin[::ti] #ti=0 is excluded
        #    spin_t_shifted = spin[t::t] 

        T_windows = spin_t.shape[0]
        print('spin_t shape: ', spin_t.shape)

        for x in range(0, safe_dist + 1):
            spin_x = spin_t[:, :-x, :] if x > 0 else spin_t
            spin_x_shifted = spin_t_shifted[:, x:, :] #instead of alpha^(-x)
            X_windows = spin_x.shape[1]
            #print('spin_xt shape: ', spin_x.shape)
            Cxt_[ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=2)) / ((X_windows) * (T_windows)) #L - x
            
        for x in range(-safe_dist , 0):
            x_abs = np.abs(x)
            spin_x = spin_t[:, x_abs:, :]
            spin_x_shifted = spin_t_shifted[:, :-x_abs, :] #alpha^(-x)
            X_windows = spin_x.shape[1]
            #print('spin_xt shape: ', spin_x.shape)
            Cxt_[ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=2)) / ((X_windows) * (T_windows)) #L - x_abs
    return Cxt_


def obtaincorrxt(file_j, path):
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path,str(begin)))
    steps = min(1601,int(Sp_aj.size/(3*L))) if L==128 else min(3201,int(Sp_aj.size/(3*L))) 
    print('steps = ', steps)
    #stepcount = min(steps, 1601)

    Sp_a = np.reshape(Sp_aj, (steps,L,3))
    print('Sp_a initial shape: ' , Sp_a.shape)
    #Sp_a = Sp_a[:stepcount]
    #print('steps , step-jump factor = ', stepcount, fine_res)
    #print('Sp_a shape: ' , Sp_a.shape)

    mconsv = calc_mconsv_ser(Sp_a, alpha)
    mconsv_tot = np.sum(mconsv, axis=1)
    Cxt = calc_Cxt_optimized(steps, mconsv, alpha) #mconsv

    print('shape of the array Cxt/Cnnxt: \n', Cxt.shape) #Cxt/Cnnxt, energ_1/energ_eta1 respectively
    return Cxt#, mconsv_tot #Cxt instead of Cxt[1:], Cxt[0] initially had zero values due to index mismatching

if L==1024: Cxtpath = f'Cxt_series_storage/L{L}/{epstr[epss]}'
else: Cxtpath = 'Cxt_series_storage/L{}'.format(L)

Cxtpath = f'Cxt_series_storage/lL{L}'
"""
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)
"""

# nconf = len(range(begin,end)) #we do not divide by nconf till the very last step - at plotting

#Mtotpath = f'H_total_storage/L{L}'

for conf in range(begin, end):
    print(conf)
    Cxt = obtaincorrxt(conf, path_to_directree) #, mconsv_tot
    if param =='xpa2b0' or param == 'qwa2b0' or param == 'xphsbg':
        f = open(f'./{Cxtpath}/{hiddensubfolder}/Cxt_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy','wb');
        np.save(f, Cxt);
        f.close()

        """g = open(f'./{Mtotpath}/alpha_ne_pm1/Mconsv_tot_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy', 'wb')
        np.save(g, mconsv_tot)
        g.close()"""

print('processing time = ', time.perf_counter() - start)

