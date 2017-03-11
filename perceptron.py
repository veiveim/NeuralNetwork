from numpy import *
import matplotlib.pyplot as plt

# input vertors
inputs = [[matrix([1,1]), matrix([0,2]), matrix([3,1])],\
      [matrix([2,-1]),matrix([2,0]), matrix([1,-2])],\
      [matrix([-1,2]),matrix([-2,1]), matrix([-1,1])]]

# truth output, [1,1] means class 1, [1,-1] means class 2, [-1,1] means class 3
t = [mat([[1],[1]]), mat([[1],[-1]]), mat([[-1],[1]])]

# weight matrix, bias matrix, output, error
W = mat([[1,1], [1,1]])
b = mat([[1],[1]])
a = mat([[1],[1]])
e = mat([[1],[1]])

# learning rate alpha, torelate value
alpha = 0.8
delta = 1
# output of nnet: a = W*p + b
 

# transfer function
def transFunc(a):
    n = mat([[0],[0]])
    if a[0,0] > delta:
        n[0,0] = 1
    elif a[0,0] < -delta:
        n[0,0] = -1
    else:
        n[0,0] = 0

    if a[1,0] > delta:
        n[1,0] = 1
    elif a[1,0] < -delta:
        n[1,0] = -1
    else:
        n[1,0] = 0
    return n


# get X,Y array of input points, when to draw the input point
def getXY(input):
    x = []
    y = []
    for i in range(len(input)):
        x.append(input[i][0,0])
        y.append(input[i][0,1])
    return [x,y]

# get X,Y array of endpoints of classfy boundary, when to draw the boundary
def getEndPoint(m,n,b):
    if n == 0:
        yleft = -4
        yright = 4
    else:
        yleft = (float(m) * 4 - b) / n
        yright = (-float(m) * 4 - b) / n

    if m == 0:
        xbottom = -4
        xtop = 4
    else:
        xbottom = (float(n) * 4 - b) / m
        xtop = (-float(n) * 4 - b) / m

    endX = []
    endY = []

    if yleft >= 4:
        endX.append(xtop)
        endY.append(4)
    elif yleft <= -4:
        endX.append(xbottom)
        endY.append(-4)
    else:
        endX.append(-4)
        endY.append(yleft)
    
    if yright >= 4:
        endX.append(xtop)
        endY.append(4)
    elif yright <= -4:
        endX.append(xbottom)
        endY.append(-4)
    else:
        endX.append(4)
        endY.append(yright)

    return [endX,endY]

   
if __name__ == '__main__':
    stop = 0
    iterate = 0
    while not stop:
        stop = 1
        iterate += 1
        for i in range(len(inputs)):
            for p in inputs[i]:
                # classfy
                a = (W * (p.T) + b)
                e = t[i] - transFunc(a)

                # learning
                W = W + alpha*e*p
                b = b + alpha*e

                # won't stop learning once an input point is misclassfied
                if e[0] != 0 or e[1] != 0:
                    stop = 0

    print("Iteration times =", iterate)
    print("\nW is")
    print(W)
    print("\nb is")
    print(b)

    # set the graph range and plot labels
    plt.axis([-4,4,-4,4])

    title = "learning rate alpha=%.2f, delta=%.2f" %(alpha, delta)
    plt.title(title)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    
    # draw the input points
    xy = getXY(inputs[0])
    plt.plot(xy[0], xy[1],'ro')
    xy = getXY(inputs[1])
    plt.plot(xy[0], xy[1],'yo')
    xy = getXY(inputs[2])
    plt.plot(xy[0], xy[1],'bo')
    
    # draw the classfy boundary
    endXY = getEndPoint(W[0,0], W[0,1], b[0,0])
    plt.plot(endXY[0],endXY[1], color='k', marker='')
    endXY = getEndPoint(W[1,0], W[1,1], b[1,0])
    plt.plot(endXY[0],endXY[1], color='k', marker='')

    # add the annotation for W, b, iteration count
    stringW = "W= %.2f  %.2f\n      %.2f  %.2f" %(W[0,0],W[0,1],W[1,0],W[1,1])
    plt.annotate(stringW, xy=(0,0), xytext=(-3.5,-3))

    stringb = "b= %.2f\n      %.2f" %(b[0,0],b[1,0])
    plt.annotate(stringb, xy=(0,0), xytext=(-1.2,-3))

    stringb = "iteration times= %d" %(iterate)
    plt.annotate(stringb, xy=(0,0), xytext=(-3.5,-2))
    
    plt.show()













