import numpy as np
from matplotlib import pyplot as plt


nprocs=np.array([1, 2, 4, 8, 16, 20])

filename = 'scaling.txt'
x = np.loadtxt(filename, comments="#", delimiter=",", unpack=False)


fig=plt.figure()

z = x/x[0]
s = 1/z

plt.plot(nprocs,s, '-o')
plt.plot(nprocs, nprocs, '--r')
plt.xlabel('N.of processors')
plt.ylabel('Speed-up')
plt.legend(loc='upper left')
plt.title('Strong scaling of model selection')
plt.show()

figname = 'strong_scaling_model_selection' + '.png'
fig.savefig(figname, dpi=fig.dpi)
