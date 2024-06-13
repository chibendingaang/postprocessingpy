from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
import numpy as np
from scipy import integrate

x = np.array([1,2,3,4,4.5,5,6,7,8])
y = x**2
y[4] += 101

import matplotlib.pyplot as plt

xx = np.linspace(1,8,51)
plt.plot(xx, CubicSpline(x,y)(xx), '--', label='spline')
plt.plot(xx, Akima1DInterpolator(x,y)(xx), '--', label='Akima1D')
plt.plot(xx, PchipInterpolator(x,y)(xx), '--', label='pchip')
plt.plot(x,y,'o')
plt.legend()
plt.show()

f = y
X = x

auc = integrate.simpson(f, X)
print(auc)
