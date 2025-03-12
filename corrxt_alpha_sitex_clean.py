
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
hidden = int(sys.argv[9])

N_array = np.concatenate(np.arange([0]), (np.power(2, np.arange(0,7)), np.arange([96, 112, 120, 124, 126, 127])))
hiddensubfolder =  'alpha_ne_pm1/waveN' if hidden in (-1)*N_array else 'alpha_ne_pm1'

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
fine_res = 5
file_src = f'{path_to_directree}/outa.txt'

def calc_Cxt_N(alpha, fine_res_param = 5):
    #spin = spin[0:steps:fine_res]
    alpha_ = alpha 
    T_array = np.concatenate((np.arange(0,100,5), np.arange(100, 250, 10),np.arange(250,1250, 25),np.arange(1250,1400,10),  np.arange(1400, 1501, 5)))
    T = T_array.shape[0]
    
    #if hidden>=0: # no drift in the correlator definition
    #    safe_dist_r = L//2; safe_dist_l = -L//2      
    if alpha_ > 1: N_array = N_array[7:]
    elif alpha_>0 and alpha_<1: N_array = N_array[:8]
    N = N_array.shape[0]

    Cxt_n = np.zeros((N,T, L))
    t_l = fine_res
    print('T_array shape: ', T)
    print('Cxt shape: ', Cxt_n.shape)

    
    f = open(file_src)
    spin_list = np.array([name for name in f.read().splitlines()])[:60]
    f.close()
    f_last = spin_list[-1]
    
    for n in N_array:
        safe_dist_r = L-n-1
        safe_dist_l = -n
        print('left/right site limits: ', safe_dist_l, safe_dist_r)

        for src in spin_list: #conf in range(begin, end):
            #print(conf)
            Sp_aj = np.loadtxt(f'{path_to_directree}/{src}')
            steps = int((Sp_aj.size/(3*L)))
            print('steps = ', steps)

            Sp_a = np.reshape(Sp_aj, (steps, L,3))
            #print('Sp_a initial shape: ' , Sp_a.shape)
            spin = Sp_a
            
            for ti, t in enumerate(T_array):
                """
                depending on our choice, we can keep # of T_windows fixed to some value
                or as in the following case, fewer windows T-separation increases
                """
                #time_screening(fine_res) is 5, but T_array, the times at which this is calculated, are T_array.shape[0]
                spin_t = spin[0:-t:t_l] if t>0 else spin[::t_l] #t < T_array[-1] else spin[:-t_l:t_l]
                spin_t_shifted = spin[t::t_l]
                #else: #no point in taking less samples when the processing time advantage is negligible 
                # spin_t = spin[0:-t:t] # if ti>0 else spin[::ti] #ti=0 is excluded
                # spin_t_shifted = spin[t::t] 
                
                T_windows = spin_t.shape[0]
                if T_windows==0: print('T_windows ', T_windows); break
              
                for x in range(safe_dist_l, safe_dist_r +1):
                    spin_prod = (spin_t[:,n]*spin_t_shifted[:,n+x])*alpha**(-2*n-x)
                    Cxt_n[n, ti, x] = spin_prod/T_windows
                if np.isnan(Cxt_n).any(): print('NaN value encountered') ; break
    return Cxt_n

CxtN = calc_Cxt_N(alpha) #mconsv
print('shape of the array Cxt/Cnnxt: \n', CxtN.shape) #Cxt/Cnnxt, energ_1/energ_eta1 respectively
#NOTE: do not divide by nconf in the corrxt_calculation stage, only when plotting/reading the average values over configurations divide by the nconfigs
#/nconf #, mconsv_tot #Cxt instead of Cxt[1:], Cxt[0] initially had zero values due to index mismatching

Cxtpath = f'Cxt_series_storage/lL{L}'
f = open(f'./{Cxtpath}/{hiddensubfolder}/CxtN_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}.npy','wb')
np.save(f, CxtN);        
f.close()

'''
for src in spin_list: #conf in range(begin, end):
    print(conf)
    Cxt = obtaincorrxt(conf, path_to_directree) #, mconsv_tot
    if param =='xpa2b0' or param == 'qwa2b0' or param == 'xphsbg':
        f = open(f'./{Cxtpath}/{hiddensubfolder}/Cxt_nL{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy','wb');
        np.save(f, Cxt);
        f.close()

        """g = open(f'./{Mtotpath}/alpha_ne_pm1/Mconsv_tot_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf+1}config.npy', 'wb')
        np.save(g, mconsv_tot)
        g.close()"""
'''
print('processing time = ', time.perf_counter() - start)

