from numpy import *
import matplotlib.pyplot as plt

# input vertors
inputs = [[matrix([1,1]), matrix([0,2]), matrix([3,1])],\
      [matrix([2,-1]),matrix([2,0]), matrix([1,-2])],\
      [matrix([-1,2]),matrix([-2,1]), matrix([-1,1])]]

# truth output
t = [mat([[1],[1]]), mat([[1],[-1]]), mat([[-1],[1]])]

# weight matrix, bias matrix, output, error, learning rate alpha, torelate value
W = mat([[1,1], [1,1]])
b = mat([[1],[1]])
a = mat([[1],[1]])
e = mat([[1],[1]])
alpha = 1
delta = 1
# output of nnet: a = W*p + b, [1,1] means class 1, [1,-1] means class 2, [-1,1] means class 3
# 

def transFunc(a):
    n = mat([[0],[0]])
    if(a[0,0] > 0):
        n[0,0] = 1
    else:
        n[0,0] = -1

    if(a[1,0] > 0):
        n[1,0] = 1
    else:
        n[1,0] = -1
    return n

def getXY(input):
    x = []
    y = []
    for i in range(len(input)):
        x.append(input[i][0,0])
        y.append(input[i][0,1])
    return [x,y]

def getEndPoint(m,n,b):
    points = []
    
    return (-float(m)*x-b)/n

if __name__ == '__main__':
    while delta > 0.1:
        delta = 0
        for i in range(len(inputs)):
            for p in inputs[i]:
                a = (W * (p.T) + b)
                e = t[i] - transFunc(a)
                
                W = W + alpha*e*p
                b = b + alpha*e
                if(abs(e[0]) > delta):
                    delta = abs(e[0])
                if(abs(e[1]) > delta):
                    delta = abs(e[1])

    print(W)
    print(b)


    plt.axis([-4,4,-4,4])
    
    xy = getXY(inputs[0])
    plt.plot(xy[0], xy[1],'ro')
    xy = getXY(inputs[1])
    plt.plot(xy[0], xy[1],'yo')
    xy = getXY(inputs[2])
    plt.plot(xy[0], xy[1],'bo')

    y0 = getY(W[0,0],W[0,1],b[0,0],-1)
    print(y0)
    y1 = getY(W[0,0],W[0,1],b[0,0],1)
    print(y1)
    plt.plot([-1,y0],[1,y1], color='k', marker='*')

    y0 = getY(W[1,0],W[1,1],b[1,0],-1)
    y1 = getY(W[1,0],W[1,1],b[1,0],1)
    print(y1)
    plt.plot([-1,y0],[1,y1], color='k', marker='*')
    plt.show()













