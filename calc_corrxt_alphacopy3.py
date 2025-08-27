#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.set_printoptions(threshold=2561)

L = int(sys.argv[1])
dtsymb = int(sys.argv[2])
dt = 0.001 * dtsymb

Lambda, Mu = map(float, sys.argv[3:5])
begin, end = map(int, sys.argv[5:7])

epss = int(sys.argv[7])
epsilon = 10**(-1.0 * epss)
choice = int(sys.argv[8])

interval = 1
t_smoothness = int(sys.argv[9])
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
alphadeci = lambda alpha: ('0' + str(int(100 * (alpha % 1)))) if (int(100 * (alpha % 1)) < 10) else (str(int(100 * (alpha % 1))))
alpha_deci = alphadeci(alpha)
alphastr_ = lambda alpha: str(int(alpha / 1)) + 'pt' + alpha_deci
alphastr = alphastr_(alpha)

path_to_directree = f'/home/nisarg/entropy_production/mpi_dynamik/xpa2b0/L{L}/alpha_{alphastr}/{dtstr}'

epstr = {3: 'emin3', 4: 'emin4', 6: 'emin6', 8: 'emin8', 33: 'min3', 44: 'min4'}

if choice == 0:
    param = 'xp' + paramstm
    if paramstm != 'a2b0':
        path = f'./{param}/L{L}/2emin3'
        path = path_to_directree
    else:
        path = f'./{param}/L{L}/alpha_{alphastr}/2emin3'
        path = path_to_directree

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


def calc_mconsv_ser(spin, alpha):
    alpha_ser = alpha**(-np.arange(L))
    alpha_ser = alpha_ser[:, np.newaxis]
    mconsv = spin * alpha_ser
    return mconsv


def calc_Cxt(Cxt_, steps, spin, alpha):
    spin = spin[0:steps:fine_res]
    if alpha < 1:
        alpha_ = 1 / alpha
    else:
        alpha_ = alpha
    T = spin.shape[0]
    for ti, t in enumerate(range(0, steps, fine_res)):
        for x in range(-L // 2 + 1, 0):
            Cxt_[ti, x] = np.sum(np.sum((spin * np.roll(np.roll(spin, -x, axis=1), -ti, axis=0))[:(T - ti), :L - np.abs(x), :], axis=2)) / ((L - np.abs(x)) * (T - ti))
        for x in range(0, L // 2 + 1):
            Cxt_[ti, x] = np.sum(np.sum((spin * np.roll(np.roll(spin, -x, axis=1), -ti, axis=0))[:(T - ti), :L - x, :], axis=2)) / ((L - x) * (T - ti))
    return Cxt_


def obtaincorrxt(file_j, path):
    Sp_aj = np.loadtxt('{}/spin_a_{}.dat'.format(path, str(begin)))
    steps = int((Sp_aj.size / (3 * L)))
    stepcount = min(steps, 1281)
    Sp_a = np.reshape(Sp_aj, (steps, L, 3))
    Sp_a = Sp_a[:stepcount]

    r = int(1. / dt)
    j = file_j

    Cxt = np.zeros((stepcount // fine_res + 1, L + 1))
    Spnbr_a = np.roll(Sp_a, -1, axis=1)
    E_loc = (-1) * np.sum(Sp_a * Spnbr_a, axis=2)
    energ_1 = np.sum(E_loc, axis=1)
    eta_ = np.array([1, 1, 1, -1, -1, -1] * int(Sp_a.size / 6))
    eta = np.reshape(eta_, Sp_a.shape)
    Sp_ngva = Sp_a * eta
    Spnbr_ngva = Spnbr_a * eta
    Eeta_loc = (-1) * np.sum(Sp_a * Spnbr_ngva, axis=2)
    energ_eta1 = np.sum(Eeta_loc, axis=1)
    mconsv = calc_mconsv_ser(Sp_a, alpha)
    mconsv_tot = np.sum(mconsv, axis=1)
    print('mconsv_total = ', mconv_tot)
    Cxt = calc_Cxt(Cxt, stepcount, mconsv, alpha)
    return Cxt[1:] / L, mconsv_tot


if L == 1024:
    Cxtpath = f'Cxt_series_storage/L{L}/{epstr[epss]}'
else:
    Cxtpath = 'Cxt_series_storage/L{}'.format(L)
Cxtpath = f'Cxt_series_storage/lL{L}'
energypath = f'H_total_storage/L{L}'
CExtpath = 'CExt_series_storage/L{}'.format(L)

nconf = len(range(begin, end))
Mtotpath = f'H_total_storage/L{L}'

for conf in range(begin, end):
    Cxt, mconsv_tot = obtaincorrxt(conf, path_to_directree)
    if param == 'xpa2b0' or param == 'qwa2b0' or param == 'xphsbg':
        f = open(f'./{Cxtpath}/alpha_ne_pm1/Cxt_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf + 1}config.npy', 'wb')
        np.save(f, Cxt)
        f.close()
        g = open(f'./{Mtotpath}/alpha_ne_pm1/Mconsv_tot_L{L}_t_{dtstr}_jump{fine_res}_{epstr[epss]}_{param}_{alphastr}_{conf}to{conf + 1}config.npy', 'wb')
        np.save(g, mconsv_tot)
        g.close()

print('processing time = ', time.perf_counter() - start)

