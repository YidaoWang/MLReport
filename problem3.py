import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

def P(q):
    n = len(q)
    res = np.zeros(n)
    for i in range(n):
        if(q[i] < 0):
            res[i] = 0
        elif(q[i] > 1):
            res[i] = 1
        else:
            res[i]=q[i]
    return res

def K(X,y):
    n=len(y)
    k = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            k[i,j] = y[i]*y[j]*np.dot(X[i],X[j])
    return k

def objectiv_fn(w,x,y,lam):
    sum = 0.
    for i in range(len(y)):
        tmp = 1-y[i]*np.dot(w,x[i])
        sum += tmp if tmp > 0 else 0
    return sum + lam * np.dot(w,w)

def dual_fn(alpha,x,y,lam):
    return -1/(4*lam)*np.dot(alpha,np.dot(K(x,y),alpha))+np.dot(alpha,np.ones(len(alpha)))

def projected_gradient(x,y,alpha,eta=0.05,lam=1,max=1000):
    n=len(y)
    m=len(x[0])
    line_height = np.zeros(max)

    for i in range(max):
        alpha = P(alpha-eta*(1/(2*lam)*np.dot(K(x,y),alpha)-1))
        print(i+1,"========================")
        #print("alpha: ",alpha)
        w = np.zeros(m)
        for j in range(n):
            w += alpha[j]*y[j]*x[j]
        w = 1/(2*lam)*w
        print("w: ",w)
        obj = objectiv_fn(w,x,y,lam)
        dual = dual_fn(alpha,x,y,lam)
        print("f: ",obj)
        line_height[i] = np.linalg.norm(obj - dual)
        print("dual err: ",line_height[i])
            
    plt.semilogy()
    plt.plot(np.array([i+1 for i in range(max)]), line_height)
    plt.show()
    return alpha

def exam1():
    n = 40
    omega = randn(1, 1)
    noise = 0.8 * randn(n, 1)
    x = randn(n, 2)
    y = np.zeros(n)
    for i in range(n):
        y[i] = 2 * (omega * x[i, 0] + x[i, 1] + noise[i] > 0) - 1
    projected_gradient(x,y,np.zeros(n))

exam1()

