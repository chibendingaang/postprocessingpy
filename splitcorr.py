import sys

print ('give threshold - 0 for 10, 1 for 25, 2 for 100')
print ('give param - 0 for hsbg, 1 for drvn')


dtsymb = int(sys.argv[1])
tstr = f'{dtsymb}emin3'

para = ['xphsbg', 'xpdrvn', 'qwhsbg', 'qwdrvn', 'qwa2b0']
j = int(sys.argv[2])
param = para[j]
L = 1024

epss = int(sys.argv[3])
#should be 4, 6, or 8
epstr = f'emin{epss}'

f=open(f'./Cxt_storage/L{L}/qwhsbg_cxt_{epstr}_out.txt', 'r')
lines=f.readlines() #f.readlines()
result=[]

for x in lines:
    result.append(x) #.split(' ')) [7])
f.close() #f.close()

#g = open('splitdecorr{}.txt'.format(param), "w+" ) 
g = open('splitcorr{}_{}_{}.txt'.format(tstr,param,epstr), 'w+')
g.writelines(result)
g.close()

f=open(f'./Cxt_storage/L{L}/qwhsbg_cnnxt_{epstr}_out.txt', 'r')
lines=f.readlines() #f.readlines()
result=[]

for x in lines:
    result.append(x) #.split(' ')) [7])
f.close() #f.close()

#g = open('splitdecorr{}.txt'.format(param), "w+" ) 
g = open('splitetacorr{}_{}_{}.txt'.format(tstr,param,epstr), 'w+')
g.writelines(result)
g.close()
