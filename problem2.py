import numpy as np

def L1(w):
    sum = 0
    for i in w:
        sum += np.abs(i)
    return sum

def f(w, lam):
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    return np.dot(np.transpose(w-mu),np.dot(A,w-mu))+lam*L1(w)

def g(w):
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    return 2*np.dot(A, w-mu)

def ST(q, C):
    n = len(q)
    res = np.zeros(n)
    for i in range(n):
        if(q[i] >= C):
            res[i] = q[i]-C
        elif(q[i] <= -C):
            res[i] = q[i]+C
    return res

def proximal_gradient(w, eta, lam, w_t=None):
    for i in range(100):
        w = ST(w-(g(w)*eta),lam*eta)
        print("itr: ",i)
        if(w_t is not None):
            print("err: ",np.linalg.norm(w_t-w))
    return w


eta = 1/(4+np.sqrt(5))
w = np.zeros(2)

# lam = 2.
# w_t = np.array([0.82,1.09])

# lam = 4.
# w_t = np.array([0.64,0.18])

lam = 6.
w_t = np.array([0.33,0])

proximal_gradient(w,eta,lam,w_t=w_t)


