import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


seed=32947923
rng=np.random.default_rng(seed)

mx=128
my=128
m=mx*my
restart=1
maxiter=100

nnz_drop=192
nsamples=5000

D=sp.diags([rng.uniform(3,4,m)],[0],shape=(m,m))
X=sp.diags([rng.uniform(-1,1,m),rng.uniform(-1,1,m),rng.uniform(-1,1,m),rng.uniform(-1,1,m)],
        [-mx,-1,1,mx],
        shape=(m,m))
A=X+D
luA=spla.splu(A)

#Maximum eigenvalues
esmax=spla.eigs(A,k=512,return_eigenvectors=False)
#Minimum eigenvalues
esmin=spla.eigs(spla.LinearOperator((m,m),matvec=luA.solve,rmatvec=lambda x : luA.solve(x,trans="T")),k=512,return_eigenvectors=False)
esmin=1.0/esmin

es=np.array(list(esmax)+list(esmin))

plt.scatter(np.real(es),np.imag(es))
plt.savefig("eigs.svg")
