#from numba import jit
import numpy as np
import scipy as sc
import time
import torch
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"]= "10.3.0"

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(device)

P = np.array([
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [1/6,1/6,0,1/6,1/2,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1/3,0,0,1/3,1/3],
    [0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0]
    ])

#print(P.shape)
#print(P)
#print("")

pt = (np.eye(P.shape[0]) - P).transpose()

#print(pt)

#@jit(nopython=True)
def null(a, rtol=1e-4):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()


def torch_null(a, rtol=1e-4):
    A = torch.Tensor(a).to(device)
    U, S, Vh = torch.linalg.svd(A)
    v_cpu = Vh.cpu()
    v = v_cpu.numpy()
    rank = (S > rtol*S[0]).sum()
    return rank, v[rank:].T.copy()

#@jit(nopython=True)
def rref(mat,precision=0):
    #def rref(mat,precision=0,GJ=False):
    m,n = mat.shape
    p,t = precision, 1e-1**precision
    A = np.around(mat.copy(),decimals=p )
    #if GJ:
        #A = np.hstack((A,np.identity(n)))
    pcol = -1 #pivot colum
    for i in range(m):
        pcol += 1
        if pcol >= n : break
        #pivot index
        pid = np.argmax( np.abs(A[i:,pcol]) )
        #Row exchange
        A[i,:],A[pid+i,:] = A[pid+i,:].copy(),A[i,:].copy()
        #pivot with given precision
        while pcol < n and np.abs(A[i,pcol]) < t:
            pcol += 1
            if pcol >= n : break
            #pivot index
            pid = np.argmax( np.abs(A[i:,pcol]) )
            #Row exchange
            A[i,:],A[pid+i,:] = A[pid+i,:].copy(),A[i,:].copy()
        if pcol >= n : break
        pivot = float(A[i,pcol])
        for j in range(m):
            if j == i: continue
            mul = float(A[j,pcol])/pivot
            A[j,:] = np.around(A[j,:] - A[i,:]*mul,decimals=p)
        A[i,:] /= pivot
        A[i,:] = np.around(A[i,:],decimals=p)
        
    #if GJ:
    #    return A[:,:n].copy(),A[:,n:].copy()
    #else:
    return A
    

start = time.time()
rank, res = torch_null(pt)
end = time.time()
print("Elapsed time = %s" % (end - start))
print(np.abs(res))
print(rank)


start = time.time()
rank, res = null(pt)
end = time.time()
print("Elapsed time = %s" % (end - start))
print(np.abs(res))
print(rank)
"""
res = rref(pt, 6)
print("rref =", res, end="\n\n")

res2 = sc.linalg.null_space(pt, 1e-6)
print("nullspace =", res2, end="\n\n")
print(np.abs(res2))

res3 = sc.linalg.null_space(res, 1e-6)
print("nullspace res =", res3, end="\n\n")

res4 = res3[:,0]
res4 = res4/res4.sum()
print(res4.sum())
print("nullspace res4 =", res4, end="\n\n")
print(24*res4)
"""



"""
start = time.time()
res = rref(pt, 6)
rank, res5 = null(res)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

start = time.time()
res = rref(pt, 6)
rank, res5 = null(res)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

start = time.time()
res = rref(pt, 6)
rank, res5 = null(res)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


start = time.time()
rank, res5 = null(pt)
end = time.time()
print("Elapsed time = %s" % (end - start))

start = time.time()
rank, res5 = null(pt)
end = time.time()
print("Elapsed time = %s" % (end - start))
"""