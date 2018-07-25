import numpy as np
import matplotlib.pyplot as plt

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

def proximal_gradient(w, eta, lam, max=100, w_t=None):

    line_height = np.zeros(max)

    for i in range(max):
        w = ST(w-(g(w)*eta),lam*eta)
        print("========================")
        print(i+1)
        print("w: ",w)
        if(w_t is not None):
            err = np.linalg.norm(w_t-w)
            print("err: ",err)
            line_height[i] = err
            
    plt.semilogy()
    plt.plot(np.array([i+1 for i in range(max)]), line_height)
    plt.show()
    return w


eta = 1/(4+np.sqrt(5))
w = np.zeros(2)

lam = 2.
w_t = np.array([0.82,1.09])

# lam = 4.
# w_t = np.array([0.64,0.18])

# lam = 6.
# w_t = np.array([0.33,0])

proximal_gradient(w,eta,lam,w_t=w_t)


