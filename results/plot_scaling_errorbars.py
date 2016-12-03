import numpy as np
from matplotlib import pyplot as plt


nprocs=np.array([1, 2, 4, 8, 10, 16, 20])

filename = 'scaling_errors.txt'
x = np.loadtxt(filename, comments="#", delimiter=",", unpack=False)
x = x.reshape((len(nprocs),-1))

def f(t):
    return t[0]/t 

speedup =  np.apply_along_axis(f, 0, x)
m = speedup.mean(1)
err = speedup.std(1)/np.sqrt(len(nprocs))*50


fig=plt.figure()
plt.errorbar(nprocs,m,yerr=err, fmt='-o', label='measured speedup')
plt.plot(nprocs, nprocs, '--r',label='theoretical speedup limit')
plt.axhline(np.max(m), linestyle='--', color='.5')
plt.xlabel('N.of processors')
plt.ylabel('Speed-up')
plt.xlim([0,21])
plt.ylim([0,21])
plt.yticks(np.arange(0,21, 2))
plt.legend(loc='upper left')
plt.title('Strong scaling 1 Ulisse node')
plt.show()

figname = 'strong_scaling_with_errors' + '.png'
fig.savefig(figname, dpi=fig.dpi)
