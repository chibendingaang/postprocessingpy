import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.fft as sff
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy import integrate
import scipy.optimize as optimization

class DataAnalysis:
    def __init__(self, L):
        self.L = L
        self.dim3 = 3
        self.steps = 1601
        self.T, self.N = 1601, 512
        self.X = np.arange(-self.L//2, self.L//2)
        self.path = f'./L{self.L}/2emin3'
        self.fine_r = 2
        self.eps = 10
        self.j = np.random.randint(11, 32)
        
    def openfile(self, m):
        inp_file = f"{self.path}/spin_a_modulated{m+1}1{self.j}.dat"
        spina = np.loadtxt(inp_file, dtype=np.float64)
        print('Configuration file: ', inp_file)
        print(spina.shape)
        return spina
    
    def calc_dxt(self, sp_a, sp_b):
        decorr = np.sum(sp_a*sp_b, axis=2)
        dxt = np.concatenate((decorr[:,self.L//2:], decorr[:,0:self.L//2]), axis=1)
        return dxt
    
    def calc_cxt(self, sp_a):
        Cxt_ = np.zeros((self.steps//self.fine_r+1, self.L))
        for ti, t in enumerate(range(0, self.steps, self.fine_r)):
            print('t: ', t)
            for x in range(self.L):
                Cxt_[ti, x] = np.sum(np.sum((sp_a*np.roll(np.roll(sp_a, -x, axis=1), -ti, axis=0))[:-ti], axis=2))/(self.steps//self.fine_r+1 - ti)
        return Cxt_
    
    def calc_mkt(self, m):
        spina = self.openfile(m)
        sp_a = np.reshape(spina, (self.steps, self.L, self.dim3))
        Sh = sp_a.shape
        Mq_1, Mq_2, Mq_3 = np.empty((Sh[0], Sh[1])), np.empty((Sh[0], Sh[1])), np.empty((Sh[0], Sh[1]))
        T, N = Mq_3.shape

        for ti in range(0, T):
            Mq_1[ti] = sff.fft(sp_a[ti, :, 0])/N
            Mq_2[ti] = sff.fft(sp_a[ti, :, 1])/N
            Mq_3[ti] = sff.fft(sp_a[ti, :, 2])/N

        Mq_ = np.sqrt(np.abs(Mq_1)**2 + np.abs(Mq_2)**2 + np.abs(Mq_3)**2)
        print('Mq_ shape: ', Mq_.shape)
        print('|Mq(0,qm): ', np.abs(Mq_[0, m]))
        print('|Mq(200,qm): ', np.abs(Mq_[200, m]))
        return Mq_, Mq_3
    
    def threeconsecutivetrues(self, arr):
        count = 0
        arr = arr*1
        for i in range(len(arr)):
            if arr[i] == 1:
                count = count + 1
            else:
                count = 0
            if count >= 3:
                return i-2
    
    def check_areauc(self, k, inputf):

