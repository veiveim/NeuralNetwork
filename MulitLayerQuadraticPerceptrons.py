from numpy import *
import numpy as np
import matplotlib.pyplot as plt



# one input layer, one hidden layer, one output layer
# here for convientence, we don't use numpy and matrix
# process the matrix-multiplication by "hand", as only 2-dimention matrix need.

# number of hidden & output layer neurons
N_input = 2
N_hidden = 10
N_output = 1

# learning rate
alpha = 0.6

# torelate error threshold
e_threshold = 0.01

# weight matrix, bias matrix, output for hidden layer
U0 = [[1.0] * N_input] * N_hidden
V0 = [[2.0] * N_input] * N_hidden
b0 = [1.0] * N_hidden
y0 = [1.0] * N_hidden
x0 = [1.0] * N_hidden
delta0 = [1.0] * N_hidden


# weight matrix, bias matrix, output for output layer
U1 = [[1.0] * N_hidden] * N_output
V1 = [[2.0] * N_hidden] * N_output
b1 = [1.0] * N_output
y1 = [1.0] * N_output
x1 = [1.0] * N_output
delta1 = [1.0] * N_output

# random function, to generate random float matrix
def random_init(M, m, n):
    for i in range(m):
        M[i] = np.random.rand(n)

# transition function, the sigmoidal activation function
def sigmoid(y):
    return 1.0/(1+exp(-y))

# transition function's derivative
def sigmoid_p(y):    
    return sigmoid(y)*(1 - sigmoid(y))

# forward pass
def forward(input):
    global y1

    # calculate the hidden layer
    for i in range(N_hidden):
        y0[i] = 0
        for j in range(N_input):
            y0[i] += U0[i][j]*input[j]*input[j] + V0[i][j]*input[j]
        y0[i] += b0[i]
        x0[i] = sigmoid(y0[i])
    
    # calculate the output layer
    for i in range(N_output):
        y1[i] = 0
        for j in range(N_hidden):
            y1[i] += U1[i][j]*x0[j]*x0[j] + V1[i][j]*x0[j]
        y1[i] += b1[i]
        x1[i] = sigmoid(y1[i])
        
    return x1

    
# backward pass
def backward(input, t):

    # calculate the output layer first
    for i in range(N_output):
        # for the output layer, the delta = error*sigmoid_p(y1[i])
        delta1[i] = (t[i]-x1[i]) * sigmoid_p(y1[i])
        for j in range(N_hidden):
            U1[i][j] += alpha * delta1[i] * x0[j] * x0[j]
            V1[i][j] += alpha * delta1[i] * x0[j]
        b1[i] += alpha * delta1[i]

    # calculate the hidden layer
    for i in range(N_hidden):
        # for the hidden layer, the delta = ssigmoid_p(y0[i])* sum_j(delta1[j]*(2U*x0+V))
        
        delta0[i] = 0        
        for j in range(N_output):
            delta0[i] += delta1[j] * (2*U1[j][i]*x0[i] + V1[j][i])
        delta0[i] = delta0[i] * sigmoid_p(y0[i])


        for j in range(N_input):
            U0[i][j] += alpha * delta0[i] * input[j] * input[j] 
            V0[i][j] += alpha * delta0[i] * input[j]
        b0[i] += (alpha * delta0[i])

    return 0
    

# main function
if __name__ == '__main__':
   
    inputf = open('two_spiral_train.txt')

    lines = inputf.readlines()
    cnt_total = len(lines)
    cnt_right = 0
    cnt_iterate = 0
    max_e = 0

    random_init(U0, N_hidden, N_input)
    random_init(V0, N_hidden, N_input)
    random_init(b0, N_hidden, 1)
    random_init(U1, N_output, N_hidden)
    random_init(V1, N_output, N_hidden)
    random_init(b1, N_output, 1)

    print("######## Learning begin ########")
    
    while cnt_right < cnt_total:
        cnt_right = 0
        cnt_iterate += 1
        max_e = 0
        for line in lines:
            words = line.split()
            input = [float(words[0]), float(words[1])]
            t = [float(words[2])]

            # forward pass
            output = forward(input)

            #calculate total error, decide wheather update weights.
            error = 0.0
            for i in range(N_output):
                e = t - output[i]
                error += e * e / 2
                
            if abs(error) > max_e:
                max_e = abs(error)
            if abs(error) < e_threshold:
                cnt_right += 1
                continue

            # backward pass
            backward(input, t)
        
            
        #print("right_num=", cnt_right, " max_error=", max_e)
    print("######## Learning success, cnt_iterate=%d ########" % cnt_iterate)
    inputf.close()
    

    # Test
    print("######## Test begin ########")
    inputf = open('two_spiral_test.txt')
    lines = inputf.readlines()

    cnt_total = len(lines)
    cnt_right = 0
    max_e = 0
    for line in lines:
        words = line.split()
        input = [float(words[0]), float(words[1])]
        t = [float(words[2])]
        output = forward(input)

        error = 0.0
        for i in range(N_output):
            e = t - output[i]
            error += e * e / 2

        if abs(error) > max_e:
            max_e = abs(error)
        if abs(error) < e_threshold:
            cnt_right += 1

    print("cnt_total=", cnt_total, "cnt_right=", cnt_right, \
          "precision=", cnt_right/cnt_total, "max_error=", max_e)
    print("######## Test finished ########")
    inputf.close()



    # draw the plot
    # set the graph range and plot labels
    size = 4
    plt.axis([-size,size, -size,size])
    ax = plt.gca()
    ax.set_aspect(1)

    title = "learning_rate=%.2f, e_threshold=%.2f" %(alpha, e_threshold)
    plt.title(title)
    xlabel = "X axis (percision=%.4f, iterate_count=%d)" % (cnt_right/cnt_total, cnt_iterate)
    plt.xlabel(xlabel)
    plt.ylabel('Y axis')

    # draw the classification boundary, us 
    grain = 80
    for i in range(grain*2):
        for j in range(grain*2):
            point = [(grain - float(i))/grain*size, (grain - float(j))/grain*size]
            value = forward(point)
            e = 1.0 - value[0]
            if(e * e / 2 < e_threshold):
                plt.plot(point[0], point[1],'yo')

    # draw the train input points.
    inputf = open('two_spiral_train.txt')
    lines = inputf.readlines()
    for line in lines:
        words = line.split()
        e = 1.0 - float(words[2])
        if(e * e / 2 < e_threshold):
            plt.plot(float(words[0]), float(words[1]),'ko')
        else:
            plt.plot(float(words[0]), float(words[1]),'ro')

    # display the final plot.
    plt.show()
    inputf.close()




