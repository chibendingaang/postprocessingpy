import sys

print ('give threshold - 0 for 10, 1 for 25, 2 for 100')
print ('give param - 0 for hsbg, 1 for drvn')

thr = [10,25,100]
i = int(sys.argv[1])
thresh = thr[i]

dtsymb = int(sys.argv[2])
if dtsymb == 1: tstr = '1emin3'
if dtsymb == 2: tstr = '2emin3'
if dtsymb == 5: tstr = '5emin3'

para = ['qwhsbg', 'qwdrvn', 'qwa2b0', 'xphsbg', 'xpdrvn']
j = int(sys.argv[3])
param = para[j]

#epschoice = ['epss', 'epsq']
#k = int(sys.argv[4])
#epstr = epschoice[k]

f=open('./Dxt_storage/{}{}out_{}.txt'.format(param,thresh,tstr),"r")
#h = open('./Dxt_storage/decorr{}.txt'.format(param),"r")
lines=f.readlines() #f.readlines()
result=[]

for x in lines:
    result.append(x) #.split(' ')) [7])

#print([x.split(' ')[4] for x in open('./timeregister/2emin3/epsq/times25/drvn25out.txt',"r").readlines()])
f.close() #f.close()

#g = open('splitdecorr{}.txt'.format(param), "w+" ) 
g = open('splitdec{}{}_{}.txt'.format(thresh,param, tstr), 'w+')
g.writelines(result)
g.close()
