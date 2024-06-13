#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First version on Wed Oct 13 17:46:47 2021
@author: nisarg
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=2561)

class Loadfiles:
    def __init__(self):
        self.L = 512 #int(sys.argv[1])
        self.dtsymb = 2 #int(sys.argv[2])
        self.dt = 0.001*self.dtsymb
        self.Lambda, self.Mu = 0.75, 0.25 #map(float, sys.argv[3:5])
        self.begin , self.end = 1 , 501 #map(int, sys.argv[5:7])
        self.epss = 3 #int(sys.argv[7])
        self.epsilon = 10**(-1.0*self.epss)
        self.choice = 0 #bool(sys.argv[8]) #either 0 or 1; double or quad precision only; folders renamed accordingly
        self.interval = 1
        
        '''
        self.t_smoothnes =  #int(sys.argv[9]) #needed for correlator calculation; not here
        if self.dtsymb == 2: 
            self.fine_res = 1*self.t_smoothness
        if dtsymb ==1: 
            self.fine_res = 2*self.t_smoothness
        '''
        self.dtstr = f'{dtsymb}emin3'    
        self.epstr = {3: 'eps_min3', 4: 'eps_min4', 6: 'eps_min6', 8: 'eps_min8' }
        
        # IMPORTANT: path to extract the files from
        if self.choice ==0:
            if self.Lambda == 1 and self.Mu == 0: self.param = 'xphsbg'
            if self.Lambda == 0 and self.Mu == 1: self.param = 'xpdrvn'
            else: self.param = 'xpa2b0'; alpha = (Lambda - Mu)/(Lambda + Mu)
            self.path =  f'./{self.param}/L{L}/alpha_0pt{int(10*alpha)}/{self.dtstr}' #path0 #usually xpdrvn 
        elif self.choice ==1:
            if self.Lambda == 1 and self.Mu == 0: self.param = 'qwhsbg'
            if self.Lambda == 0 and self.Mu == 1: self.param = 'qwdrvn'
            else: self.param = 'qwa2b0'
            self.path = f'./L{L}/{epstr[epss]}/{self.param}' #path1  #the largest file data is currently here: 17042023
    		#usually qwdrvn #the epss = 3 for this path
        '''
        elif self.choice ==2:
            if self.Lambda == 1 and self.Mu == 0: self.param = 'hsbg'
            if self.Lambda == 0 and self.Mu == 1: self.param = 'drvn'
            if self.Lambda == 1 and self.Mu == 1: self.param = 'qwa2b0'
            self.path = f'./{self.param}/{self.dtstr}'	#path2 # the self.param is drvn or hsbg here; choice = 2
        '''
    
    def load_singlefile(self, file_j):  
        # Loads a single file
        # Standardises the number of steps for different files 
        # Introduce dat_files
        
        filerepo = f'{self.path}/dat_files.txt'
        Sp_aj = np.loadtxt('{}/{}'.format(self.path,str(self.file_j)))
        steps = int((Sp_aj.size/(3*self.L))) #0.1*dtsymb multiplication omitted
 
        #reshaped to count spin_xt at every fine_resolved time step
        Sp_a = np.reshape(Sp_aj, (steps,self.L,3)) #[0:steps:fine_res]
        print('Sp_a.shape: ' , Sp_a.shape)
        
        stepcount = min(steps, 961)
        Sp_a = Sp_a[:stepcount] 
        print('steps , step-jump factor = ', stepcount, fine_res)
        
        return Sp_a, steps
    
    def check_conserved(self, Sp_a):
        # Take the temporal array of spins and check if the defined conserved quantity
        # is indeed conserved
        # First, define that quantity: also, check for even number of sites
        
        m_consv = 0.5*(Sp_a[t, 2*x, :] + Sp_a[t, 2*x+1,:]/alpha)/(alpha**(2*x))
        
        
        
    
    
    
    def run(self):
        start  = time.perf_counter()
        
        filerepo = f'{self.path}/dat_files.txt'
        f = open(filerepo, 'r')
    	filelist = np.array([filex for filex in f.read().splitlines()]) [begin:end]
    	f.close()
    	
        for j, file_j in enumerate(filelist):
            Sp_a, steps = self.load_singlefile(file_j)
        
        end = time.perf_counter()
        print(f'Elapsed time: {end - start} seconds')
        
if __name__ == '__main__':
    loader = Loadfiles()
    loader.run()
    



