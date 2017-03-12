from numpy import *
import matplotlib.pyplot as plt


# one input layer, one hidden layer, one output layer
# here for convientence, we don't use numpy and matrix
# process the matrix-multiplication by "hand", as only 2-dimention matrix need.

# number of hidden neurons
N_hidden = 10

# learning rate
alpha = 100

# error
error = 1.0

# weight matrix, bias matrix, output for hidden layer
U0 = [[1.0,1.0]] * N_hidden
V0 = [[2.0,2.0]] * N_hidden
b0 = [1.0] * N_hidden
y0 = [1.0] * N_hidden
x0 = [1.0] * N_hidden


# weight matrix, bias matrix, output for output layer
U1 = [1.0] * N_hidden
V1 = [2.0] * N_hidden
b1 = 1.0
y1 = 1.0
x1 = 1.0

# transition function, the sigmoidal activation function
def sigmoid(y):
    return 1.0/(1+exp(-y))

# transition function's derivative
def sigmoid_p(y):
    return exp(-y)/((1+exp(-y))*(1+exp(-y)))

# forward pass
def forward(input1):
    global y1

    # as x^2 is need, calculate here
    input2 = [input1[0]*input1[0], input1[1]*input1[1]]

    # calculate the hidden layer
    for i in range(N_hidden):
        y0[i] = U0[i][0]*input2[0] + V0[i][0]*input1[0] + \
                U0[i][1]*input2[1] + V0[i][1]*input1[1] + b0[i]
        x0[i] = sigmoid(y0[i])
    
    # calculate the output layer
    y1 = b1
    for i in range(N_hidden):
        y1 += U1[i]*x0[i]*x0[i] + V1[i]*x0[i]
    x1 = sigmoid(y1)
        
    return x1

    
# backward pass
def backward(error):
    global b1
    
    # calculate the output layer first
    # for the output layer, the delta = error*sigmoid_p(y1)
    delta1 = error * sigmoid_p(y1)

    print(delta1, error)
    for i in range(N_hidden):
        U1[i] -= alpha * delta1 * x0[i] * x0[i] 
        V1[i] -= alpha * delta1 * x0[i]
    b1 -= alpha * delta1

    # calculate the hidden layer
    for i in range(N_hidden):
        # for the hidden layer, the delta = sigmoid_p(y0[i])*delta1*(2U+V)
        delta0 = sigmoid_p(y0[i]) * delta1 * (2*U1[i]*x0[i] + V1[i])
        
        U0[i] -= alpha * delta0 * x0[i] * x0[i] 
        V0[i] -= alpha * delta0 * x0[i]
        b0[i] -= alpha * delta0
        
    return 0
    

# main function
if __name__ == '__main__':
   
    inputf = open('two_spiral_train.txt')

    cnt = 0
    max_e = 0
    lines = inputf.readlines()
    while cnt < 50:
        cnt = 0
        max_e = 0
        i = 0
        for line in lines:
            input = line.split()

            i += 1
            if i > 10:
                break
        
            # forward pass
            result = forward([float(input[0]), float(input[1])])
            error = float(input[2]) - result

            if abs(error) < 0.5:
                cnt += 1
                continue

            if abs(error) > max_e:
                max_e = abs(error)
            
            # backward pass
            backward(error)
        
            
        print(cnt, max_e)
        #print(U1[0], U1[1])





