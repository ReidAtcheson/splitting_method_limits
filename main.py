import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import pickle


seed=32947923
rng=np.random.default_rng(seed)

mx=128
my=128
m=mx*my
restart=1
maxiter=100

nnz_drop=100
nsamples=1000

D=sp.diags([rng.uniform(3,4,m)],[0],shape=(m,m))
X=sp.diags([rng.uniform(-1,1,m),rng.uniform(-1,1,m),rng.uniform(-1,1,m),rng.uniform(-1,1,m)],
        [-mx,-1,1,mx],
        shape=(m,m))
A=X+D




def residual_digits(A,Ah,rng):
    m,n=A.shape
    assert(m==n)
    assert(A.shape==Ah.shape)
    b=rng.uniform(-1,1,m)
    res=[]
    def callback(xk):
        nonlocal res
        r=b-A@xk
        res.append(np.linalg.norm(r,ord=np.inf))
    luAh = spla.splu(Ah)
    spla.gmres(A,b,restart=restart,maxiter=maxiter,callback=callback,callback_type="x",M=spla.LinearOperator((m,m),matvec=luAh.solve))
    return -np.log(res[-1])/len(res)



def splitting(X,D,nnz_drop,rng):
    m,_=X.shape
    X=sp.coo_matrix(X)
    vals=X.data
    rids=X.row
    cids=X.col

    rcvs = rng.choice(list(zip(rids,cids,vals)),size=len(rids)-nnz_drop)

    rids_out=[]
    cids_out=[]
    vals_out=[]
    for r,c,v in rcvs:
        rids_out.append(r)
        cids_out.append(c)
        vals_out.append(v)



    out=sp.coo_matrix((vals_out,(rids_out,cids_out)),shape=(m,m))
    return sp.csc_matrix(out+D)





resits=[]
start=time.time()
for it in range(0,nsamples):
    Ah = splitting(X,D,nnz_drop,rng)
    resit = residual_digits(A,Ah,rng)
    resits.append(resit)
    print(resit)
    if it%100==0:
        f=open("resits.dat","wb")
        pickle.dump(resits,f)
    stop=time.time()

    time_per_it=(stop-start)/(it+1)
    print("Time remaining: ",time_per_it*(nsamples-it-1)/60," minutes")

f=open("resits.dat","wb")
pickle.dump(resits,f)




