#import sys

L = 128
dtsymb = 2
dt = 0.001*dtsymb

# lambda, mu strictly integer for Hsbg, NR model
# else, can be float
Lambda = 0.975
Mu = 0.025

# we only want static variables through this file, so using sys.argv is a No!
#begin = int(sys.argv[1]) #20
#end = int(sys.argv[2]) #40

epss = 3
epsilon = 10**(-1.0*epss)

choice = 0


