import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

uniform_data = np.random.rand(100, 112)

x, y = uniform_data.shape

X = np.arange(0,x)
Y = np.arange(0,y)
#fig, ax = plt.subplots()
plt.pcolormesh(Y,X,uniform_data,cmap='twilight')
plt.show()

#ax = sns.heatmap(uniform_data, linewidth=0.5)
#plt.savefig('./plots/snsheatmap.png')
#plt.show()
