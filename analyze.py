import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle

resits=np.array(pickle.load(open("resits.dat","rb")))
norms=np.array(pickle.load(open("norms.dat","rb")))
invnorms=np.array(pickle.load(open("invnorms.dat","rb")))

ids=list(filter(lambda i :resits[i]>0 ,range(0,len(resits))))

resits=resits[ids]
norms=norms[ids]
invnorms=invnorms[ids]


plt.hist(resits,bins=20)
plt.scatter([0.0],[1.1033258414043903],s=8)
plt.xlim([0.0,1.5])
plt.savefig("resits.svg")
plt.close()

plt.scatter(norms,resits)
plt.xlabel("norms")
plt.ylabel("resits")
plt.savefig("norms.svg")
plt.close()

plt.scatter(np.log(np.array(invnorms)),resits)
plt.xlabel("invnorms")
plt.ylabel("resits")
plt.savefig("invnorms.svg")
