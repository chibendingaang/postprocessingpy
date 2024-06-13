import numpy as np
import sys
import matplotlib.pyplot as plt
import time

class DataProcessor:
    def __init__(self, L, dtsymb, Lambda, Mu, begin, end, epss):
        self.L = L
        self.dtsymb = dtsymb
        self.Lambda = Lambda
        self.Mu = Mu
        self.begin = begin
        self.end = end
        self.epss = epss

    def read_data(self):
        # Code to read data from files
        r = int(1./dt)
    	Dxt = np.ones((2*steps+1, L), dtype=np.longdouble)
    	#Dnewxt = np.empty((steps+1,L))
    	#t = step*r
   		if self.L==1024: filepath = f'./L{self.L}/eps_min4/{param}'
    	else: filepath = f'./{param}/{dtstr}'
    	for j in filenum:
    		Sp_aj = np.loadtxt(f'{filepath}/spin_a_{str(j)}.dat')
    		Sp_bj = np.loadtxt(f'{filepath}/spin_b_{str(j)}.dat')
    		Sp_a = np.reshape(Sp_aj, (2*steps+1,L,3)); Sp_b = np.reshape(Sp_bj, (2*steps+1,L,3))           
    		Dxt  -= np.sum(Sp_a*Sp_b,axis=2)/len(filenum)
    	Dnewxt = np.concatenate((Dxt[:,L//2:], Dxt[:,0:L//2]), axis = 1) 
		return Dnewxt
        pass

    def calculate_average(self):
        # Code to calculate average
        pass

    def generate_plot(self):
        # Code to generate plot
        pass

    def process_data(self):
        start = time.perf_counter()
        self.read_data()
        self.calculate_average()
        self.generate_plot()
        stop = time.perf_counter()
        print("Time taken:", stop - start)

if __name__ == "__main__":
    L = int(sys.argv[1])
    dtsymb = int(sys.argv[2])
    Lambda = int(sys.argv[3])
    Mu = int(sys.argv[4])
    begin = int(sys.argv[5])
    end = int(sys.argv[6])
    epss = int(sys.argv[7])

    processor = DataProcessor(L, dtsymb, Lambda, Mu, begin, end, epss)
    processor.process_data()

