import numpy as np
import matplotlib.pyplot as plt

#begin = 4
#end = 320

xlist= dict()
xlist[0] = 0
xlist[1] = 32
xlist[2] = 64
xlist[3] = 96

np.set_printoptions(threshold=2500)

#repo of all the Dxt_array outputs
filerepo = 'outDxtmin4.txt' 
filerepo2 = 'outDxtmin6.txt'
#these store all the filenames of Dxtarray_files
filepath = 'Dxt_storage/L1024'

'''
for filex in files_:
    tmp = np.loadtxt(f'./{filepath}/{filex.split()[0]}')
    Dxtfiles.append(filex.split()[0])
    #print(tmp.shape)
''' 


fig, (ax1, ax2) = plt.subplots(2,figsize=(14,14))

for j in range(1,4):
    site = xlist[j]
    
    f = open(f'./{filepath}/{filerepo}')
    files_ = f.readlines()
    f.close()
    totalfiles = len(files_); 
    filespersite = totalfiles//len(xlist) 	#must be an integer, used later for indexing

    Dxt = np.loadtxt(f'./{filepath}/{files_[0].split()[0]}')
    Dxtavg_4 = np.zeros(Dxt.shape)
    Dxtavg_6 = np.zeros(Dxt.shape)
    Dxt4_site0 = []
    Dxt6_site0 = []

    for filex in files_[j*filespersite:(j+1)*filespersite]:
        Dxt_4 = np.loadtxt(f'./{filepath}/{filex.split()[0]}') 
        #Dxt_{}_array_L1024_t_1emin3_qwhsbg_eps_min4_{}to{}config.txt') #.format(site,conf, conf+1))
        if j ==0: Dxt4_site0.append(Dxt_4[100]); 
        Dxtavg_4 += Dxt_4

    f = open(f'./{filepath}/{filerepo2}')
    files_ = f.readlines()
    f.close()
    totalfiles = len(files_); filespersite2 = totalfiles//len(xlist)
    print('no. of files per site for Dxt_min4, Dxt_min6 respectively: '); print(filespersite, filespersite2)

    for filex in files_[j*filespersite2:(j+1)*filespersite2]:
        Dxt_6 = np.loadtxt(f'./{filepath}/{filex.split()[0]}') 
        if j ==0: Dxt6_site0.append(Dxt_6[100])
        Dxtavg_6 += Dxt_6
    
    Dxtavg_4 = Dxtavg_4[40:]/filespersite 	#len(range(begin+40,end+1))
    Dxtavg_6 = Dxtavg_6[40:]/filespersite2 	#len(range(begin+40,end+1))
    #print(Dxtavg)
    print('Max Dxt value at site=0, t = 100 for epsilon 10^(-4), 10^(-6) cases respectively : ')
    
    if Dxt4_site0: print(max(Dxt4_site0))	#Doesn't print array if it is None
    if Dxt6_site0: print(max(Dxt6_site0), '\n')

    t_range = np.arange(len(Dxtavg_4)-60)/10
    logDxtavg_4 = np.log(Dxtavg_4*10**8)[60:]
    
    for i in range(len(logDxtavg_4)):
        logDxtavg_4[i] = 5*logDxtavg_4[i]/i 
        #10/(2*t); factor of 10 due to collecting dt at every 10th t-step    
    ax1.plot(t_range,logDxtavg_4,label=f'{xlist[j]}')
    
    logDxtavg_6 = np.log(Dxtavg_6*10**12)[60:]
    for i in range(len(logDxtavg_6)):
        logDxtavg_6[i] = 5*logDxtavg_6[i]/i
    
    ax2.plot(t_range,logDxtavg_6,label=f'{xlist[j]}')

plt.suptitle(r"$\varepsilon = 10^{-4}$ vs $\varepsilon = 10^{-6}$ ")
ax1.set_ylim(-1.25,2.5)
ax2.set_ylim(-1.25,2.5)
ax1.grid(); ax2.grid()
ax1.legend(); ax2.legend()
plt.savefig('./plots/logDxt_few{}config.png'.format(filespersite))
#plt.plot(Dxt)
#plt.title('Dxt, site = {}'.format(site))
#plt.savefig('./plots/Dxt_fewtosingleconfig.png')

