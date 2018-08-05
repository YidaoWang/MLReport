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

def proximal_gradient(w, g, eta, lam, w_t, max=50):
    line_height = np.zeros(max)
    print("repeat count, error")

    for i in range(max):
        w = ST(w-(g(w)*eta),lam*eta)
        err = np.linalg.norm(w_t-w)
        print("{0}, {1}".format(i+1,err))
        line_height[i] = err
            
    return line_height

def accelerated_proximal_gradient(w, g, eta, lam, w_t, max=50):
    print("repeat count, error")
    line_height = np.zeros(max)
    s = 1.
    v = w
    for i in range(max):
        w1=ST(v-(g(v)*eta),lam*eta)
        
        err = np.linalg.norm(w_t-w1)
        print("{0}, {1}".format(i+1,err))
        line_height[i] = err

        s1=(1+np.sqrt(1+4*s*s))/2.
        qt = (s-1)/s1
        s=s1

        v=w1+qt*(w1-w)
        w=w1
            
    return line_height

def exam1():
    eta = 1/(4+np.sqrt(5))
    w = np.array([3,-1])
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    g = lambda w:2*np.dot(A, w-mu)
    LAMBDAS = [2., 4., 6.]
    W_OPTS = np.array([[0.82, 1.09], [0.64, 0.18], [0.33, 0]])

    for i in range(0,3):
        lam = LAMBDAS[i]
        w_t = W_OPTS[i]
        line = proximal_gradient(w,g,eta,lam,w_t)
        plt.plot(np.array([i+1 for i in range(max)]), line)
        plt.semilogy()
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        plt.savefig('problem2_1_{0}.png'.format(i+1))
        plt.clf()

def exam2():
    eta = 1/(4+np.sqrt(5))
    w = np.array([3,-1])
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    g = lambda w:2*np.dot(A, w-mu)
    LAMBDAS = [2., 4., 6.]
    W_OPTS = np.array([[0.82, 1.09], [0.64, 0.18], [0.33, 0]])

    for i in range(0,3):
        lam = LAMBDAS[i]
        w_t = W_OPTS[i]
        line = accelerated_proximal_gradient(w,g,eta,lam,w_t)
        plt.plot(np.array([i+1 for i in range(max)]), line)
        plt.semilogy()
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        plt.savefig('problem2_1_{0}.png'.format(i+1))
        plt.clf()

def compare():
    eta = 1/(4+np.sqrt(5))
    w = np.array([3,-1])
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([1,2])
    g = lambda w:2*np.dot(A, w-mu)
    max=50
    LAMBDAS = [2., 4., 6.]
    W_OPTS = np.array([[0.82, 1.09], [0.64, 0.18], [0.33, 0]])
    lines = []
    for i in range(0,3):
        lam = LAMBDAS[i]
        w_t = W_OPTS[i]
        lines.append(plt.plot(np.array([j+1 for j in range(max)]), proximal_gradient(w,g,eta,lam,w_t), label='PG(λ={0})'.format(lam)))
    
    for i in range(0,3):
        lam = LAMBDAS[i]
        w_t = W_OPTS[i]
        lines.append(plt.plot(np.array([j+1 for j in range(max)]), accelerated_proximal_gradient(w,g,eta,lam,w_t),label='APG(λ={0})'.format(lam)))
  
    plt.legend()
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.savefig('problem2.png')
    plt.clf()


#exam1()
#exam2()
compare()