import numpy as np
import matplotlib.pyplot as plt

def L1(w):
    sum = 0
    for i in w:
        sum += np.abs(i)
    return sum

def ST(q, C):
    n = len(q)
    res = np.zeros(n)
    for i in range(n):
        if(q[i] >= C):
            res[i] = q[i]-C
        elif(q[i] <= -C):
            res[i] = q[i]+C
    return res

def proximal_gradient(w, g, eta, lam, max=30, w_t=None):

    line_height = np.zeros(max)

    for i in range(max):
        w = ST(w-(g(w)*eta),lam*eta)
        print(i+1,"========================")
        print("w: ",w)
        if(w_t is not None):
            err = np.linalg.norm(w_t-w)
            print("err: ",err)
            line_height[i] = err
            
    plt.semilogy()
    plt.plot(np.array([i+1 for i in range(max)]), line_height)
    plt.show()
    return w

def accelerated_proximal_gradient(w, g, eta, lam, max=30, w_t=None):
    line_height = np.zeros(max)
    s1 = 1.
    for i in range(max):
        s=s1
        w1=ST(w-(g(w)*eta),lam*eta)
        s1=(1+np.sqrt(1+4*s*s))/2.
        w=w1+((s-1)/s1)*(w1-w)
        print(i+1,"=========================")
        print("w: ",w)
        if(w_t is not None):
            err = np.linalg.norm(w_t-w)
            print("err: ",err)
            line_height[i] = err
            
    plt.semilogy()
    plt.plot(np.array([i+1 for i in range(max)]), line_height)
    plt.show()
    return w

def exam1():
    eta = 1/(4+np.sqrt(5))
    w = np.array([3,-1])
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    g = lambda w:2*np.dot(A, w-mu)
    # lam = 2.
    # w_t = np.array([0.82,1.09])
    lam = 4.
    w_t = np.array([0.64,0.18])
    # lam = 6.
    # w_t = np.array([0.33,0])
    proximal_gradient(w,g,eta,lam,w_t=w_t)

def exam2():
    eta = 1/(4+np.sqrt(5))
    w = np.array([3,-1])
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    g = lambda w:2*np.dot(A, w-mu)
    # lam = 2.
    # w_t = np.array([0.82,1.09])
    lam = 4.
    w_t = np.array([0.64,0.18])
    # lam = 6.
    # w_t = np.array([0.33,0])
    accelerated_proximal_gradient(w,g,eta,lam,w_t=w_t)

exam2()