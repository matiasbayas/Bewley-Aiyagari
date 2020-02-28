import numpy as np
import scipy as sp
from scipy.stats import norm
from numba import vectorize, guvectorize, njit
import RognlieCode as rc


def agrid(amin, amax, I):
    
    a = np.linspace(amin, amax, I)    
    
    return a


def updatev(V, a, y, La, r, gamma, rho, Delta = 0.01):
    
    amin = a[0]
    da = a[2] - a[1]
    I = np.shape(a)[0]
    J = np.shape(y)[0]
    dVf = np.zeros([I,J])
    dVb = np.zeros([I,J])
       
    dVf[0:I-1,:] = (V[1:I,:] - V[0:I-1,:])/da
    dVf[I-1,:] = 0
    dVb[1:I,:] = (V[1:I,:] - V[0:I-1,:])/da
    dVb[0,:] = (y[np.newaxis,:] + r*amin)**(-gamma)

    i_v = dVb > dVf

    cf = dVf**(-1/gamma)
    sf = np.transpose(y[:,np.newaxis] + r*a) - cf
    cb = dVb**(-1/gamma)
    sb = np.transpose(y[:,np.newaxis] + r*a) - cb
    c0 = np.transpose(y[:,np.newaxis] + r*a)
    dV0 = c0**(-gamma)

    iF = sf > 0
    iB = sb < 0
    i0 = 1 - iF - iB
    iF[I-1,:] = 0
    iB[I-1,:] = 1

    dV = dVf*iF + dVb*iB + dV0*i0
    c = dV**(-1/gamma)
    s = np.transpose(y[:,np.newaxis] + r*a) - c
    h = c**(1-gamma)/(1-gamma) + s*dV

    Vi = (h - rho*V + V@La)*Delta + V
   
    return Vi, s, c


def improvev(V, s, c, a, y, La, r, gamma, rho, Delta = 0.01, it = 150):
    
    amin = a[0]
    da = a[2] - a[1]
    I = np.shape(a)[0]
    J = np.shape(y)[0]
    dVf = np.zeros([I,J])
    dVb = np.zeros([I,J])
    
    iF = s > 0
    iB = s < 0
    i0 = 1 - iF - iB
    
    c0 = np.transpose(y[:,np.newaxis] + r*a)
    dV0 = c0**(-gamma)

    for j in range(it):
        dVf[0:I-1,:] = (V[1:I,:] - V[0:I-1,:])/da
        dVf[I-1,:] = 0
        dVb[1:I,:] = (V[1:I,:] - V[0:I-1,:])/da
        dVb[0,:] = (y[np.newaxis,:] + r*amin)**(-gamma)
        dV = dVf*iF + dVb*iB + dV0*i0
        h = c**(1-gamma)/(1-gamma) + s*dV
        V = (h - rho*V + V@La)*Delta + V
        
        
    return V



def solvehh(V, a, y, La, r, gamma, rho, Delta = 0.01, tol = 1e-8, maxit = 20_000):
    
    for it in range(maxit):
        Vi, s, c = updatev(V, a, y, La, r, gamma, rho)
        if it % 5 == 0 and rc.within_tolerance(V, Vi, tol):
            break
        V = Vi
        
        V = improvev(V, s, c, a, y, La, r, gamma, rho)
        
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
        
    return V, s, c

@njit
def forwarditerateD(D, La, s, Delta, da):
    
    I = D.shape[0]
    J = D.shape[1]

    # First, create the transition matrix
    Tr = Delta*s/da
    TrR = np.maximum(Tr,0)
    TrL = -np.minimum(Tr,0)

    Dnew = np.zeros_like(D)
    Dnew[I,J] = D[I,J] 
    
    for i in range(I-1):
        for j in range(J):
            d = D[i,j]
            di = D[i+1,j]
            Dnew[i, j] += d + di*TrL[i+1,j] - d*TrR[i,j]
            Dnew[i+1, j] += d*TrR[i,j] - di*TrL[i+1,j]

    Dnew = Dnew + D@(np.transpose(Delta*La))

    return Dnew

def solveD(D_seed, La, s, Delta, da, tol=1E-10, maxit=50_000):

    D = D_seed

    for it in range(maxit):
        Dnew = forwarditerateD(D, La, s, Delta, da)

        # only check convergence every 5 iterations for efficiency
        if it % 5 == 0 and rc.within_tolerance(D, Dnew, tol):
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    D = Dnew

    return D



