#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
#from mpi4py import MPI
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.set_printoptions(threshold=2561)


# MPI setup
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

# Read input arguments
L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 * dtsymb

Lambda, Mu = map(float, sys.argv[3:5])
begin, end = map(int, sys.argv[5:7])

epss = int(sys.argv[7])
epsilon = 10 ** (-1.0 * epss)
choice = int(sys.argv[8])

interval = 1
t_smoothness = int(sys.argv[9])  # typically? 100
if dtsymb == 2:
    fine_res = 1 * t_smoothness
if dtsymb == 1:
    fine_res = 2 * t_smoothness

dtstr = f'{dtsymb}emin3'
epstr = {3: 'eps_min3', 4: 'eps_min4', 6: 'eps_min6', 8: 'eps_min8'}

if Lambda == 1 and Mu == 0:
    paramstm = 'hsbg'
elif Lambda == 0 and Mu == 1:
    paramstm = 'drvn'
else:
    paramstm = 'a2b0'

alpha = (Lambda - Mu) / (Lambda + Mu)
alpha = (Lambda - Mu)/(Lambda + Mu)
alphadeci = lambda alpha: ('0' + str(int(100*(alpha%1)))) if (int(100*(alpha%1)) < 10) else (str(int(100*(alpha%1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha/1)) + 'pt' + alpha_deci 
alphastr = alphastr_(alpha)
print('alphastr = ', alphastr)

epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

if choice == 0:
    param = 'xp' + paramstm
    if paramstm != 'a2b0':
        path = f'./{param}/L{L}/2emin3'
    else:
        path = f'./{param}/L{L}/alpha_{alphastr}/2emin3'
if choice == 1:
    param = 'qw' + paramstm
    if paramstm != 'a2b0':
        path = f'./{param}/L{L}/2emin3'
    else:
        param = 'qwa2b0'
        path = f'./{param}/2emin3/alpha_{alphastr}'
if choice == 2:
    if paramstm == 'a2b0':
        param = 'qw' + paramstm
        path = f'./{param}/2emin3/alpha_{alphastr}'
    else:
        param = paramstm
        path = f'./{param}/L{L}/2emin3'

start = time.perf_counter()

def calc_Cxt(Cxt_, steps, spin):
    for ti, t in enumerate(range(0, steps, fine_res)):
        for x in range(L):
            Cxt_[ti, x] = np.sum(np.sum((spin * np.roll(np.roll(spin, -x, axis=1), -ti, axis=0))[:-ti], axis=2)) / (steps // fine_res + 1 - ti)
    return Cxt_

def obtaincorrxt(file_j, path):
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path, file_j))
    steps = int((Sp_aj.size / (3 * L)))
    Sp_a = np.reshape(Sp_aj, (steps, L, 3))
    stepcount = min(steps, 521) #512 -- 521 -- 641 -- 961
    Sp_a = Sp_a[:stepcount]
    r = int(1. / dt)
    j = file_j
    Cxt = np.zeros((stepcount // fine_res + 1, L))
    CExt = np.zeros((stepcount // fine_res + 1, L))
    Spnbr_a = np.roll(Sp_a, -1, axis=1)
    E_loc = (-1) * np.sum(Sp_a * Spnbr_a, axis=2)
    energ_1 = np.sum(E_loc, axis=1)
    eta_ = np.array([1, 1, 1, -1, -1, -1] * int(Sp_a.size / 6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a * eta
    Spnbr_ngva = Spnbr_a * eta
    Eeta_loc = (-1) * np.sum(Sp_a * Spnbr_ngva, axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis=1)
    mconsv = np.zeros((stepcount, L, 3))
    for ti in range(mconsv.shape[0]):
        for x in range(mconsv.shape[1]):
            mconsv[ti, x] = (Sp_a[ti, x]) / (alpha) ** x
    def mconsv_mdecay(param):
        if param == 'qwhsbg' or param == 'xphsbg' or param == 'hsbg':
            return Sp_a, Sp_ngva
        if param == 'qwdrvn' or param == 'xpdrvn' or param == 'drvn':
            return Sp_ngva, Sp_a
        if param == 'xpa2b0' or param == 'qwa2b0':
            return mconsv
    mconsv = mconsv_mdecay(param)
    Cxt = calc_Cxt(Cxt, stepcount, mconsv)
    return Cxt[1:] / L

Cxtpath = 'Cxt_series_storage/L{}'.format(L)
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)

'''
# Split work among processors
nconf = end - begin
confs_per_proc = nconf // size
extra = nconf % size

# Determine the start and end indices for each process
local_begin = begin + rank * confs_per_proc + min(rank, extra)
local_end = local_begin + confs_per_proc + (1 if rank < extra else 0)
'''
for conf in range(begin, end): #local_begin, local_end):
    #if (rank == conf - local_begin):
    Cxt = obtaincorrxt(conf, path)
    if param == 'xpa2b0' or param == 'qwa2b0':
        f = open(f'./{Cxtpath}/alpha_ne_pm1/Cxt_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf + 1}config.npy', 'wb')
        np.save(f, Cxt)
        f.close()

#comm.Barrier()
#if rank == 0:
#    print('Processing time =', time.perf_counter() - start)

