#written by ColonelFazackerley
from matplotlib import pyplot as mp
import numpy as np

def peak(x, c, sigmasq):
    return np.exp(-np.power(x - c, 2) / sigmasq)

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    #np.where(condition, [x,y]) -> x if condition True; y if False
    #eqv to np.nonzero(arr) -> True for nonzero elements
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

# make some fake data
x=np.linspace(0,20,21)
y1=peak(x,10,16)
y2 = peak(x,10,4)
y3 = peak(x,10,64)

# find the two crossing points
# a convincing plot
for y in [y1,y2,y3]:
    hmx = half_max_x(x,y)
    print(hmx)

    # print the answer
    fwhm = hmx[1] - hmx[0]
    print("FWHM:{:.3f}".format(fwhm))
    half = max(y)/2.0
    mp.plot(x,y)
    mp.plot(hmx, [half, half])
mp.show()
