import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, guvectorize, njit

def agrid(amax,N,amin=0,pivot=0.25):
    """Grid with a+pivot evenly log-spaced between
    amin+pivot and amax+pivot.
    """
    a = np.geomspace(amin+pivot,amax+pivot,N) - pivot
    a[0] = amin # make sure *exactly* equal to amin
    return a


@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for same-dim x1, x2"""
    # implement by obtaining flattened views using ravel, then looping
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True



def updatev(V,a,y,La,r,gamma,rho,Delta = 0.01):
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

def hh_iter(v_seed, r, gamma, rho, a_grid, y, La, Delta = 0.01, tol = 1e-8, maxit = 20_000):

    V = v_seed
    a = a_grid
    amin = a[0]
    da = a[2] - a[1]
    I = np.shape(a)[0]
    J = np.shape(y)[0]
    dVf = np.zeros([I,J])
    dVb = np.zeros([I,J])
    c = np.zeros([I,J])

    for it in range(maxit):

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

        if it % 5 == 0 and within_tolerance(V, Vi, tol):
            break
        V = Vi

        for j in range(100):
            dVf[0:I-1,:] = (V[1:I,:] - V[0:I-1,:])/da
            dVf[I-1,:] = 0
            dVb[1:I,:] = (V[1:I,:] - V[0:I-1,:])/da
            dVb[0,:] = (y[np.newaxis,:] + r*amin)**(-gamma)
            dV = dVf*iF + dVb*iB + dV0*i0
            h = c**(1-gamma)/(1-gamma) + s*dV
            V = (h - rho*V + V@La)*Delta + V

    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    return V, s, c


@njit
def forward_iterate2(D, La, s, Delta, da):

    # First, create the transition matrix
    Tr = Delta*s/da
    TrR = np.maximum(Tr,0)
    TrL = -np.minimum(Tr,0)

    Dnew = np.zeros_like(D)
    for i in range(D.shape[0]-1):
        for j in range(D.shape[1]):
            d = D[i,j]
            di = D[i+1,j]
            Dnew[i, j] += d + di*TrL[i+1,j] - d*TrR[i,j]
            Dnew[i+1, j] += d*TrR[i,j] - di*TrL[i+1,j]

    Dnew = Dnew + D@(np.transpose(Delta*La))

    return Dnew

def dist_ss2(D_seed, La, s, Delta, da, tol=1E-10, maxit=50_000):

    D = D_seed



    for it in range(maxit):
        Dnew = forward_iterate2(D, La, s, Delta, da)

        # only check convergence every 5 iterations for efficiency
        if it % 5 == 0 and within_tolerance(D, Dnew, tol):
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    D = Dnew

    return D/da
