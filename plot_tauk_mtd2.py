import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sff
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema

def check_x_intercept(func):
    x_intercepts = []
    for i, fi in enumerate(func[0:-1]):
        if fi*func[i+1] < 0: x_intercepts.append(i)
    return x_intercepts

def threeconsecutiveones(arr):
    count = 0
    for i in range(len(arr)):
        if arr[i] == 1: count += 1
        else: count = 0
        if count >= 3:
            return i-2; break

def localextrema(arr):
    loc_max = argrelextrema(arr, np.greater)
    loc_min = argrelextrema(arr, np.less)
    return loc_max, loc_min

#def tau_mthd3(func):
    #the steady state method:
    #use the tau_methods 1 and 2 to get the tau_relaxation
    #then find the steady state tau through the correlator



def tau_mthd2(func):
    #this method involves using localextrema to mask the input func
    #and then checking if the three consecutive maxima are decreasing
    #we next find the index of the maxima where func[t] <= cutoff * func[0]
    print("Sp_3[t=0]: ", func[0])
    cutoff = np.exp(-4) 
    print("cutoff*Sp3[0] = ", cutoff*func[0])
    loc_max = localextrema(func)[0]
    loc_max = np.array(loc_max).flatten()
    mask_cutoff = func[loc_max] < cutoff*func[0] 
    return loc_max, func[loc_max], mask_cutoff


def tau_mthd1(func):
    #this function interpolates input function func to f1 through CubicSpline
    #two methods for tau: 
    
    #method1: just find five consecutive x-intersections, and ensure that 3 consecutive peaks are reducing
    #         the local peaks lie between 0 and x1, x2 and x3, x4 and x5
    #to find x-intercept: check when f1(x1).f2(x2) <0 and take x_intercept = (x1+x2)*0.5
    f = CubicSpline(T, func)(T)    
    t_intercepts = check_x_intercept(f)
    if len(t_intercepts) >= 4:
        tau = t_intercepts[4]
        return T[tau]
    else: 
        print('Tau cannot be found since function damped out quickly')

T = np.arange(0,20.005, 0.01)
#func = np.exp(-T)*np.cos(np.pi*T/8)
func = np.exp(-T)*np.cos(np.pi*T/2)
tau = tau_mthd1(func)
print(tau)

tau2 = tau_mthd2(func)
print(tau2)
    
     
