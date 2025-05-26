#!/usr/bin/python3

"""
suggested update, #29-03-2025:
just try once with alpha^(-x) instead of alpha^(-x -2n) 
to see if Subroto's reflection speculation was relevant only due to the (suppressed) power coefficient!
"""

import os
import sys
print(sys.executable)
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

import params
sys.path.append('/Users/nisargq/Library/Python/3.9/bin')
np.set_printoptions(threshold=2561)


class CorrxtAnalyzer:
    def __init__(self, begin, end):
        self.L = params.L
        self.dtsymb = params.dtsymb
        self.dt = params.dt
        self.Lambda = lamda #params.Lambda
        self.Mu = mu #params.Mu
        self.begin = begin
        self.end = end
        self.epss = params.epss
        self.epsilon = params.epsilon
        self.choice = params.choice

        self.alpha = (self.Lambda - self.Mu) / (self.Lambda + self.Mu)
        self.fine_res = fine_res  #5
        self.epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

        self.alpha_deci = self._get_alpha_deci(self.alpha)
        self.alphastr = self._get_alphastr(self.alpha)
        self.param = self._label_param()

        self.hiddensubfolder = 'alpha_ne_pm1'

        self.path_to_directree = f'/home/nisarg/entropy_production/mpi_dynamik/xpa2b0/L{self.L}/alpha_{self.alphastr}/{self._get_dtstr()}'

        self.Cxtpath = f'Cxt_series_storage/L{self.L}' #lL{self.L}
        self.N_array = self._compute_N_array()
        self.T_array = self._get_T_array()#[:56] if self.L==64 else self._get_T_array()
    
    
    def _get_T_array(self):
        if self.L == 64 or self.L==128:
            return np.arange(0, 801, 5)
        elif self.L == 192 or self.L==256:
            return np.arange(0, 1201, self.fine_res)
        else:
            return np.concatenate((np.arange(0, 100, 5), np.arange(100, 250, 10), \
                np.arange(250, 1250, 25), np.arange(1250, 1400, 10), \
                np.arange(1400, 1501, 5)))
    
    def _compute_N_array(self):
        raw = np.concatenate((np.arange(2,33,2), np.arange(96,128,2)))
        #np.concatenate((([0]), np.array([1,2,4,8]), np.arange(16, 65, 8),np.arange(80,128,16),  self.L - np.array([8,4,2,1])))
        #np.concatenate([0], np.power(2, np.arange(0, 7)), 128 - np.power(2, np.arange(5,-1,-1)))
        if self.alpha > 1:
            return raw[:16]
        elif 0 < self.alpha < 1:
            return raw[17:]
        return raw

    def _get_alpha_deci(self, alpha):
        #if alpha>=0:
        
        #deci = int(100 * (alpha % 1))
        #return ('0' + str(deci)) if deci < 10 else str(deci)
        #just use np.abs(alpha)%1 
        #if alpha<=0:
        deci = int(100 * (np.abs(alpha) % 1))
        deci_th = int(1000*(np.abs(alpha)%1))<10
        if alpha < 1 and deci_th:
                return '975'
        return ('0' + str(deci)) if deci < 10 else str(deci)


    def _get_alphastr(self, alpha):
        if alpha>0:
            return str(int(alpha / 1)) + 'pt' + self.alpha_deci
        if alpha<0:
            return 'min_' + str(int(np.abs(alpha) / 1)) + 'pt' + self.alpha_deci

    def _get_dtstr(self):
        return f'{self.dtsymb}emin3'

    def _label_param(self):
        if self.Lambda == 1 and self.Mu == 0:
            paramstm = 'hsbg'
        elif self.Lambda == 0 and self.Mu == 1:
            paramstm = 'drvn'
        else:
            paramstm = 'a2b0'

        if self.choice == 0:
            param = 'xp' + paramstm
            #path = self.path_to_directree

        elif self.choice == 1:
            param = 'qw' + paramstm
            if paramstm == 'a2b0':
                param = 'qwa2b0'
                #path = f'./{param}/2emin3/alpha_{self.alphastr}'

        elif self.choice == 2:
            if paramstm == 'a2b0':
                param = 'qw' + paramstm
                #path = f'./{param}/2emin3/alpha_{self.alphastr}'
            else:
                param = paramstm
                #path = f'./{param}/L{self.L}/2emin3'

        print('param, alphastr: ', param, self.alphastr)
        return param

    #@profile
    def calc_mconsv_Hconsv(self, spin): 
        # multiplies the spin series and neighbouring spin product series with alpha^{-x} term 
        alpha_ser = self.alpha ** (-np.arange(self.L))[:, np.newaxis]
        mconsv = spin * alpha_ser
        spin_dot = np.sum(spin[:, :-1, :] * spin[:,1:, :], axis = -1)  # shape (L-1, spin.shape[1])
        print('spin, spin_prod, alpha_ser shapes: ', spin.shape, spin_dot.shape, alpha_ser.shape)
        
        alpha_ser2 = self.alpha ** (-np.arange(self.L))#[:, None, None] 
        # we need 1-D series here, not 3-D!
        Hconsv = spin_dot * alpha_ser2[1:]  # slice the weights for x = 1 to L-1
        #Hconsv = spin_dot * alpha_ser[1:, 0, 0][:, None] 
        #print('Hconsv beyond t=26: ', Hconsv.shape, '\n', Hconsv[26:-1:5])
        Htotal = np.sum(Hconsv, axis=-1)
        #print('Htotal :',Htotal.shape, '\n', Htotal[::5])
        return mconsv, Hconsv, Htotal

    #@profile
    def calc_Cxt_optimized(self, spin, en_dens):
        # spin and energy density arrays
        steps = spin.shape[0]

        T = self.T_array.shape[0] 

        Cxt_ = np.zeros((T, self.L + 1))
        Cxt_energy = np.zeros((T, self.L + 1))
        print('Cxt.shape: ', Cxt_.shape)

        safe_dist = self.L // 2
        t_l = self.fine_res

        for ti, t in enumerate(self.T_array):
            spin_t = spin[0:-t:t_l] if t > 0 else spin[::t_l]
            spin_t_shifted = spin[t::t_l]
            endens_t = en_dens[0:-t:t_l] if t > 0 else en_dens[::t_l]
            endens_t_shifted = en_dens[t::t_l]
            T_windows = spin_t.shape[0]

            for x in range(0, safe_dist + 1):
                # Make sure the memory requirement is still satisfied even when the array grows large
                spin_x = spin_t[:, :-x, :] if x > 0 else spin_t
                spin_x_shifted = spin_t_shifted[:, x:, :]
                endens_x = endens_t[:, :-x] if x>0 else endens_t
                endens_x_shifted = endens_t_shifted[:, x:]

                X_windows = spin_x.shape[1]
                Cxt_[ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=1)) / (X_windows * T_windows)
                Cxt_energy[ti, x] = np.sum(np.sum(endens_x*endens_x_shifted, axis=-1))/(X_windows * T_windows)

            for x in range(-safe_dist, 0):
                x_abs = np.abs(x)
                spin_x = spin_t[:, x_abs:, :]
                spin_x_shifted = spin_t_shifted[:, :-x_abs, :]
                endens_x = endens_t[:, x_abs:] 
                endens_x_shifted = endens_t_shifted[:, :-x_abs]
                X_windows = spin_x.shape[1]
                Cxt_[ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=1)) / (X_windows * T_windows)
                Cxt_energy[ti, x] = np.sum(np.sum(endens_x * endens_x_shifted, axis=-1)) / (X_windows * T_windows)

        return Cxt_, Cxt_energy


    def calc_Cxt_site_indexed(self, fine_res_param=5):
        T = self.T_array.shape[0]
        N = self.N_array.shape[0]

        Cxt_n = np.zeros((N, T, self.L))
        file_src = f'{self.path_to_directree}/outa.txt'
        t_l = fine_res_param

        print('T_array shape: ', T)
        print('Cxt shape: ', Cxt_n.shape)

        with open(file_src) as f:
            spin_list = np.array([name for name in f.read().splitlines()])[self.begin:self.end]

        for ni, n in enumerate(self.N_array):
            safe_dist_r = self.L - n - 1
            safe_dist_l = -n
            print('left/right site limits: ', safe_dist_l, safe_dist_r)

            for src in spin_list:
                Sp_aj = np.loadtxt(f'{self.path_to_directree}/{src}')
                steps = int(Sp_aj.size / (3 * self.L))
                Sp_a = np.reshape(Sp_aj, (steps, self.L, 3))
                spin = Sp_a

                for ti, t in enumerate(self.T_array):
                    spin_t = spin[0:-t:t_l] if t > 0 else spin[::t_l]
                    spin_t_shifted = spin[t::t_l]
                    T_windows = spin_t.shape[0]
                    if T_windows == 0:
                        print('T_windows ', T_windows)
                        break

                    for x in range(safe_dist_l, safe_dist_r + 1):
                        spin_prod = np.sum(np.sum(spin_t[:, n, :] * spin_t_shifted[:, n + x, :] * self.alpha ** (-2 * n - x), axis=1))
                        Cxt_n[ni, ti, x] = spin_prod / T_windows

                    if np.isnan(Cxt_n).any():
                        print('NaN value encountered')
                        break

        return Cxt_n

    def calc_Cxt_time_indexed(self, fine_res_param=5):
        # T-array is kept local for the Cxt-time-indexed function 
        #N_array = np.concatenate((([0]), np.array([1,2,4,8]), np.arange(16, 65, 8),np.arange(80,128,16),  self.L - np.array([8,4,2,1])))
        #T_array = np.arange(0, 1501, 5) #steps = 1601
        T = self.T_array.shape[0]
        N = self.N_array.shape[0]

        #Cxt_n = np.zeros((N, T, self.L))
        Cxt_ncopy = np.zeros((N, T, self.L))
        file_src = f'{self.path_to_directree}/outa.txt'
        t_l = fine_res_param

        print('T_array shape: ', T)
        print('Cxt shape: ', Cxt_ncopy.shape)

        with open(file_src) as f:
            spin_list = np.array([name for name in f.read().splitlines()])[self.begin:self.end]

        for ni, n in enumerate(self.N_array):
            safe_dist_r = self.L - n - 1
            safe_dist_l = -n
            print('left/right site limits: ', safe_dist_l, safe_dist_r)

            for src in spin_list:
                Sp_aj = np.loadtxt(f'{self.path_to_directree}/{src}')
                steps = int(Sp_aj.size / (3 * self.L))
                Sp_a = np.reshape(Sp_aj, (steps, self.L, 3))
                spin = Sp_a[0::t_l]
                #spin_t = spin[0::t_l] if t > 0 else spin[::t_l]
                T_windows = spin.shape[0]
                if T_windows == 0:
                    print('T_windows ', T_windows)
                    break

                for ti, t in enumerate(self.T_array):
                    #spin_t_shifted = spin[t::t_l]
                    for x in range(safe_dist_l, safe_dist_r+1):
                        spin_n = spin[0,n,:]
                        spin_n_shifted = spin[ti,n+x,:]
                        Cxt_ncopy[ni,ti,x] = np.sum(spin_n*spin_n_shifted)
                    '''
                    for x in range(0, safe_dist_r + 1):
                        spin_x = spin[0, :-x, :] if x > 0 else spin
                        spin_x_shifted = spin[ti, x:, :]
                        X_windows = spin_x.shape[1]
                        Cxt_n[ni, ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=1)) / (X_windows)
                        """ New change:
                        Cxt_ncopy[ni,ti,x] = np.sum(spin[0,n,:]*spin[ti,n+x,:], axis=1)

                        """

                    for x in range(safe_dist_l, 0):
                        x_abs = np.abs(x)
                        spin_x = spin[0, x_abs:, :]
                        spin_x_shifted = spin[ti, :-x_abs, :]
                        X_windows = spin_x.shape[1]
                        Cxt_n[ni, ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=1)) / (X_windows)
                    '''
                    if np.isnan(Cxt_ncopy).any(): #Cxt_n
                        print('NaN value encountered')
                        break

        return Cxt_ncopy #Cxt_n

    def calc_Cxt_time_indexed_full_timerange(self, fine_res_param=5):
        # T-array is kept local for the Cxt-time-indexed function 
        #N_array = np.concatenate((([0]), np.array([1,2,4,8]), np.arange(16, 65, 8),np.arange(80,128,16),  self.L - np.array([8,4,2,1])))
        T_array = np.arange(0, 1501, 5) #steps = 1601
        T = T_array.shape[0]
        N = self.N_array.shape[0]

        Cxt_n = np.zeros((N, T, self.L))
        #Cxt_ncopy = np.zeros((N, T, self.L))
        file_src = f'{self.path_to_directree}/outa.txt'
        t_l = fine_res_param

        print('T_array shape: ', T)
        print('Cxt shape: ', Cxt_n.shape)

        with open(file_src) as f:
            spin_list = np.array([name for name in f.read().splitlines()])[self.begin:self.end]

        for ni, n in enumerate(self.N_array):
            safe_dist_r = self.L - n - 1
            safe_dist_l = -n
            print('left/right site limits: ', safe_dist_l, safe_dist_r)

            for src in spin_list:
                Sp_aj = np.loadtxt(f'{self.path_to_directree}/{src}')
                steps = int(Sp_aj.size / (3 * self.L))
                Sp_a = np.reshape(Sp_aj, (steps, self.L, 3))
                spin = Sp_a[0::t_l]
                #spin_t = spin[0::t_l] if t > 0 else spin[::t_l]
                T_windows = spin.shape[0]
                if T_windows == 0:
                    print('T_windows ', T_windows)
                    break

                for ti, t in enumerate(T_array):
                    #spin_t_shifted = spin[t::t_l]
                    for x in range(safe_dist_l, safe_dist_r+1):
                        spin_n = spin[0,n,:]
                        spin_n_shifted = spin[ti,n+x,:]
                        Cxt_n[ni,ti,x] = np.sum(spin_n*spin_n_shifted* self.alpha ** (-2 * n - x))

                    if np.isnan(Cxt_n).any(): #Cxt_n
                        print('NaN value encountered')
                        break

        return Cxt_n
    
    def calc_Cxt_time_indexed_unweighted(self, fine_res_param=5):
        # T-array is kept local for the Cxt-time-indexed function 
        #N_array = np.concatenate((([0]), np.array([1,2,4,8]), np.arange(16, 65, 8),np.arange(80,128,16),  self.L - np.array([8,4,2,1])))
        T_array = np.arange(0, 1501, 5) #steps = 1601
        T = T_array.shape[0]
        N = self.N_array.shape[0]

        Cxt_n = np.zeros((N, T, self.L))
        #Cxt_ncopy = np.zeros((N, T, self.L))
        file_src = f'{self.path_to_directree}/outa.txt'
        t_l = fine_res_param

        print('T_array shape: ', T)
        print('Cxt shape: ', Cxt_n.shape)

        with open(file_src) as f:
            spin_list = np.array([name for name in f.read().splitlines()])[self.begin:self.end]

        for ni, n in enumerate(self.N_array):
            safe_dist_r = self.L - n - 1
            safe_dist_l = -n
            print('left/right site limits: ', safe_dist_l, safe_dist_r)

            for src in spin_list:
                Sp_aj = np.loadtxt(f'{self.path_to_directree}/{src}')
                steps = int(Sp_aj.size / (3 * self.L))
                Sp_a = np.reshape(Sp_aj, (steps, self.L, 3))
                spin = Sp_a[0::t_l]
                #spin_t = spin[0::t_l] if t > 0 else spin[::t_l]
                T_windows = spin.shape[0]
                if T_windows == 0:
                    print('T_windows ', T_windows)
                    break

                for ti, t in enumerate(T_array):
                    #spin_t_shifted = spin[t::t_l]
                    for x in range(safe_dist_l, safe_dist_r+1):
                        spin_n = spin[0,n,:]
                        spin_n_shifted = spin[ti,n+x,:]
                        Cxt_n[ni,ti,x] = np.sum(spin_n*spin_n_shifted)

                    if np.isnan(Cxt_n).any(): #Cxt_n
                        print('NaN value encountered')
                        break

        return Cxt_n


    def save_Cxt_site_indexed(self, Cxt_n):
        filename = f'./{self.Cxtpath}/{self.hiddensubfolder}/wave_timeslider_N/CxtN_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                   f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{self.begin}to{self.end}configth_proximalbndry.npy'
        with open(filename, 'wb') as f:
            np.save(f, Cxt_n)
        print(f'CxtN saved to: {filename}')

    def save_Cxt_time_indexed(self, Cxt_n):
        filename = f'./{self.Cxtpath}/{self.hiddensubfolder}/wave_siteslider_N/CxtN_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                   f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{self.begin}to{self.end}configth_proximalbndry.npy'
        with open(filename, 'wb') as f:
            np.save(f, Cxt_n)
        print(f'CxtN saved to: {filename}')

    def save_Cxt_time_indexed_full_timerange(self, Cxt_n):
        filename = f'./{self.Cxtpath}/{self.hiddensubfolder}/wave_siteslider_N/full_timerange/CxtN_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                   f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{self.begin}to{self.end}configth_proximalbndry.npy'
        with open(filename, 'wb') as f:
            np.save(f, Cxt_n)
        print(f'CxtN saved to: {filename}')
    
    #@profile
    def obtaincorrxt(self, conf):
        h5file_path = f'spin_trajectories_L{self.L}_alpha_{self.alphastr}.h5'  # <-- centralized HDF5 file
        with h5py.File(h5file_path, 'r') as f:
            groupname = f'/run_{conf}/trajectory' #trajectory, not spins
            #groupname = f'/run_{conf:3d}/spins'  # 0-padded run index (0000, 0001, etc.)
            Sp_a = f[groupname][...]  # Read full dataset into memory

        print('Sp_a initial shape: ', Sp_a.shape)

        mconsv, Hconsv, Htotal = self.calc_mconsv_Hconsv(Sp_a)
        Cxt, Cxt_energy = self.calc_Cxt_optimized( mconsv, Hconsv) #Sp_a.shape[0],

        print('shape of the array Cxt/Cnnxt: ', Cxt.shape)
        return Sp_a, Cxt, Cxt_energy, Htotal
        
    def obtaincorrxt_old(self, conf):
        """
        file_path = f'{self.path_to_directree}/spin_a_{self.begin}.dat'
        Sp_aj = np.loadtxt(file_path)
        steps = min(1601, int(Sp_aj.size / (3 * self.L))) if self.L == 128 else min(3201, int(Sp_aj.size / (3 * self.L)))

        Sp_a = np.reshape(Sp_aj, (steps, self.L, 3))"""

        h5file_path = f'spin_trajectories_L{self.L}_alpha_{self.alphastr}.h5'  # <-- centralized HDF5 file
        with h5py.File(h5file_path, 'r') as f:
            groupname = f'/run_{conf:04d}/spins'  # 0-padded run index (0000, 0001, etc.)
            Sp_a = f[groupname][...]  # Read full dataset into memory

        print('Sp_a initial shape: ', Sp_a.shape)

        mconsv, Hconsv, Htotal = self.calc_mconsv_Hconsv(Sp_a)
        Cxt, Cxt_energy = self.calc_Cxt_optimized(mconsv, Hconsv) #steps = spin.shape[0]
        print('shape of the array Cxt/Cnnxt: ', Cxt.shape)
        return Sp_a, Cxt, Cxt_energy

    #@profile
    def process_all_configs(self):
        h5file_path = f'spin_trajectories_L{self.L}_alpha_{self.alphastr}.h5'
        start = time.perf_counter()

        with h5py.File(h5file_path, 'r') as f:
            for conf in range(self.begin, self.end):
                print(conf)
                groupname = f'/run_{conf}/trajectory'
                #groupname = f'/run_{conf:04d}/spins'
                if groupname not in f:
                    print(f'Warning: {groupname} not found in file')
                    continue

                Sp_a = f[groupname][...]
                print('Sp_a shape:', Sp_a.shape)

                mconsv, Hconsv, Htotal = self.calc_mconsv_Hconsv(Sp_a)
                Cxt, Cxt_energy = self.calc_Cxt_optimized(mconsv, Hconsv) #steps = Sp_a.shape[0]
                

                if self.param in ['xpa2b0', 'qwa2b0', 'xphsbg']:
                    out_file = f'./{self.Cxtpath}/{self.hiddensubfolder}/Cxt_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                                f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                    with open(out_file, 'wb') as f_out:
                        np.save(f_out, Cxt)

                    out_file2 = f'./{self.Cxtpath}/{self.hiddensubfolder}/Cxt_energy_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                                f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                    with open(out_file2, 'wb') as f_out:
                        np.save(f_out, Cxt_energy)
                    
                    out_file3 = f'./{self.Cxtpath}/{self.hiddensubfolder}/total_energy_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                                f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                    with open(out_file3, 'wb') as f_out:
                        np.save(f_out, Htotal)
                    
                    # self.Magxt
                    """out_spinfile = f'./Magxt_series_storage/Mxt_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                                f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                    with open(out_spinfile, 'wb') as f_out:
                        np.save(f_out, Sp_a)
                    out_magfile = f'./Magxt_series_storage/Malphaxt_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                                f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                    with open(out_magfile, 'wb') as f_out:
                        np.save(f_out, mconsv)
                    """
        print('processing time = ', time.perf_counter() - start)


    def process_all_configs_old(self):
        start = time.perf_counter()

        for conf in range(self.begin, self.end):
            print(conf)
            Cxt, Cxt_energy = self.obtaincorrxt(conf)[-1]
            if self.param in ['xpa2b0', 'qwa2b0', 'xphsbg']:
                out_file = f'./{self.Cxtpath}/{self.hiddensubfolder}/Cxt_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                           f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                with open(out_file, 'wb') as f:
                    np.save(f, Cxt)
                out_file = f'./{self.Cxtpath}/{self.hiddensubfolder}/Cxt_L{self.L}_t_{self._get_dtstr()}_jump{self.fine_res}_' \
                           f'{self.epstr[self.epss]}_{self.param}_{self.alphastr}_{conf}to{conf + 1}config.npy'
                with open(out_file, 'wb') as f:
                    np.save(f, Cxt)

        print('processing time = ', time.perf_counter() - start)


if __name__ == '__main__':
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    fine_res = int(sys.argv[3])
    lamda = float(sys.argv[4])
    mu = float(sys.argv[5])
    full_tmrng = 0 if end-begin==50 else 1
    analyzer = CorrxtAnalyzer(begin, end)
    analyzer.process_all_configs()
    #if full_tmrng == 0:
    #    CxtN_t = analyzer.calc_Cxt_time_indexed()
    #    analyzer.save_Cxt_time_indexed(CxtN_t)
    #elif full_tmrng == 1:
    #    CxtN_t = analyzer.calc_Cxt_time_indexed_full_timerange()
    #    analyzer.save_Cxt_time_indexed_full_timerange(CxtN_t)

