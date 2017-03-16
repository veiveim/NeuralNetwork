from numpy import *
import numpy as np
import matplotlib.pyplot as plt


# As we may need multiple MLQP, we define the MLQP class here
# one input layer, one hidden layer, one output layer
# here for convientence, we don't use numpy and matrix
# process the matrix-multiplication by "hand", as only 2-dimention matrix need.

class MLQP:    
    # number of hidden & output layer neurons
    N_input = 2
    N_hidden = 10
    N_output = 1

    # input and true value matrix for input layer
    input = []
    t = []

    # weight matrixs, local gradient matrix and output matrix for hidden layer
    U0 = [[]]
    V0 = [[]]
    b0 = []
    y0 = []
    x0 = []
    delta0 = []

    # weight matrixs, local gradient matrix and output matrix for output layer
    U1 = [[]]
    V1 = [[]]
    b1 = []
    y1 = []
    x1 = []
    delta1 = []
    
    # learning rate
    alpha = 0.1
    # torelate error threshold
    e_threshold = 0.01

    # iteraion count for training
    cnt_iterate = 0

    def __init__(self, n_input, n_hidden, n_output):
        # number of hidden & output layer neurons
        self.N_input = n_input
        self.N_hidden = n_hidden
        self.N_output = n_output

        # input matrix, true value matrix
        self.input = [0] * n_input
        self.t = [0] * n_output
        
        # weight matrix, bias matrix, output for hidden layer
        self.U0 = [[]] * n_hidden
        self.V0 = [[]] * n_hidden
        self.y0 = [0] * n_hidden
        self.x0 = [0] * n_hidden
        self.delta0 = [0] * n_hidden

        # init weight matrix with random number
        for i in range(n_hidden):
            self.U0[i] = np.random.rand(n_input)
            self.V0[i] = np.random.rand(n_input)
        self.b0 = np.random.rand(n_hidden)

        # weight matrix, bias matrix, output for output layer
        self.U1 = [[]] * n_output
        self.V1 = [[]] * n_output
        self.y1 = [0] * n_output
        self.x1 = [0] * n_output
        self.delta1 = [0] * n_output

        # init weight matrix with random number
        for i in range(n_output):
            self.U1[i] = np.random.rand(n_hidden)
            self.V1[i] = np.random.rand(n_hidden)
        self.b1 = np.random.rand(n_output)


    # set input and true value
    def add_case(self, p, tvalue):
        for i in range(self.N_input):
            self.input[i] = p[i]
        for i in range(self.N_output):
            self.t[i] = tvalue[i]
    

    # transition function, the sigmoidal activation function
    def sigmoid(self, y):
        return 1.0/(1+exp(-y))


    # transition function's derivative
    def sigmoid_p(self, y):    
        return self.sigmoid(y)*(1 - self.sigmoid(y))


    # forward pass
    def forward(self):
        # calculate the hidden layer
        for i in range(self.N_hidden):
            self.y0[i] = 0
            for j in range(self.N_input):
                self.y0[i] += self.U0[i][j]*self.input[j]*self.input[j] + \
                              self.V0[i][j]*self.input[j]
            self.y0[i] += self.b0[i]
            self.x0[i] = self.sigmoid(self.y0[i])
        
        # calculate the output layer
        for i in range(self.N_output):
            self.y1[i] = 0
            for j in range(self.N_hidden):
                self.y1[i] += self.U1[i][j]*self.x0[j]*self.x0[j] + \
                              self.V1[i][j]*self.x0[j]
            self.y1[i] += self.b1[i]
            self.x1[i] = self.sigmoid(self.y1[i])
        return self.x1

    
    # backward pass
    def backward(self):
        # update the output layer first
        for i in range(self.N_output):
            # for the output layer, the local gradient delta = error*sigmoid_p(y1[i])
            self.delta1[i] = (self.t[i]-self.x1[i]) * self.sigmoid_p(self.y1[i])

            for j in range(self.N_hidden):
                self.U1[i][j] += self.alpha * self.delta1[i] * self.x0[j] * self.x0[j]
                self.V1[i][j] += self.alpha * self.delta1[i] * self.x0[j]
            self.b1[i] += self.alpha * self.delta1[i]

        # update the hidden layer
        for i in range(self.N_hidden):
            # for the hidden layer, the local gradient delta =
            # ssigmoid_p(y0[i])* sum_j(delta1[j]*(2U*x0+V))
            self.delta0[i] = 0        
            for j in range(self.N_output):
                self.delta0[i] += self.delta1[j] * (2*self.U1[j][i]*self.x0[i] + self.V1[j][i])
            self.delta0[i] = self.delta0[i] * self.sigmoid_p(self.y0[i])

            for j in range(self.N_input):
                self.U0[i][j] += self.alpha * self.delta0[i] * self.input[j] * self.input[j] 
                self.V0[i][j] += self.alpha * self.delta0[i] * self.input[j]
            self.b0[i] += (self.alpha * self.delta0[i])
        return 0


    # training
    def train(self, train_set):     
        cnt_total = len(train_set)
        cnt_right = 0
        self.cnt_iterate = 0
        max_e = 0
        
        while cnt_right < cnt_total:
            cnt_right = 0
            self.cnt_iterate += 1
            max_e = 0
            for line in train_set:
                words = line.split()
                input = [float(words[0]), float(words[1])]
                t = [float(words[2])]
                self.add_case(input, t)

                # forward pass
                output = self.forward()

                #calculate total error, decide wheather update weights.
                error = 0.0
                for i in range(self.N_output):
                    e = t - output[i]
                    error += e * e / 2
                if abs(error) > max_e:
                    max_e = abs(error)
                    
                # only if the output close to the true value enough, we think it's right.
                if abs(error) < self.e_threshold:
                    cnt_right += 1
                    continue

                # backward pass
                self.backward()
            
            #print("right_num=", cnt_right, " max_error=", max_e)
        print("Learning success, cnt_iterate=%d\t########" % self.cnt_iterate)


    # test
    def test(self, test_set):      
        output = []
        for line in test_set:
            words = line.split()
            input = [float(words[0]), float(words[1])]
            t = [float(words[2])]
            self.add_case(input, t)
            
            result = self.forward()
            for i in range(self.N_output):
                if result[i] < 0.5:
                    output.append(0)
                else:
                    output.append(1)

        return output

    

# main function
if __name__ == '__main__':
    # init MLQP
    n_sub_part = 2
    n_input = 2
    n_hidden = 10
    n_output = 1    
    MLQPs = [MLQP(n_input, n_hidden, n_output)]
    for i in range(n_sub_part * n_sub_part - 1):
        MLQPs.append(MLQP(n_input, n_hidden, n_output))

    # train
    inputf = open('two_spiral_train.txt')
    lines = inputf.readlines()
    input00 = []
    input01 = []
    input10 = []
    input11 = []

    # randomly decompose the problem into 4 sub-problems
    # first divide each class of input vector into 2 sub-parts.
    for line in lines:
        words = line.split()
        coin = random.randint(0,99)
        # class 0
        if abs(float(words[2])) < 0.5:
            if coin % 2 == 0:
                input00.append(line)
            else:
                input01.append(line)
        # class 1
        else:
            if coin % 2 == 0:
                input10.append(line)
            else:
                input11.append(line)

    # combine sub-parts of class 0 and class 1 to create sub-problems.
    train_sets = [input00 + input10, input00 + input11, \
                  input01 + input10, input01 + input11]

    # train each MLQP.
    for i in range(n_sub_part * n_sub_part):
        print("######## MLQP %d Learning begin..." % i, end=' ')
        MLQPs[i].train(train_sets[i])
    inputf.close()
    

    # test
    print("######## Test begin...", end=' ')
    inputf = open('two_spiral_test.txt')
    lines = inputf.readlines()
    output = []
    
    # test each MLQP
    for i in range(n_sub_part * n_sub_part):
        output.append(MLQPs[i].test(lines))

    # combine outputs of each MLQP to create final output.
    i = 0
    cnt_right = 0
    for line in lines:
        words = line.split()
        t = float(words[2])
        x = output[0][i] and output[1][i] or output[2][i] and output[3][i]
        e = t - float(x)
        if e*e/2 < MLQPs[0].e_threshold:
            cnt_right += 1
        i += 1
    inputf.close()
    print("Test finished, precision=%.4f\t########" % (float(cnt_right)/len(lines)))



    # draw the plot
    # set the graph range and plot labels
    size = 4
    plt.figure("Min-Max")
    plt.axis([-size,size, -size,size])
    ax = plt.gca()
    ax.set_aspect(1)
    
    title = "learning_rate=%.2f, e_threshold=%.2f" %(MLQPs[0].alpha, MLQPs[0].e_threshold)
    plt.title(title)
    xlabel = "X axis (percision=%.4f)" % (cnt_right/len(lines))
    plt.xlabel(xlabel)
    plt.ylabel('Y axis')
    
    print("Painting Min-Max plot...")

    # draw the classification boundary
    grain = 80
    for i in range(grain*2):
        for j in range(grain*2):
            point = [(grain - float(i))/grain*size, (grain - float(j))/grain*size]
            output = []
            for k in range(n_sub_part * n_sub_part):
                MLQPs[k].add_case(point, [0])
                output.append(MLQPs[k].forward())

            value = (output[0][0] > 0.5) and (output[1][0] > 0.5) or \
                    (output[2][0] > 0.5) and (output[3][0] > 0.5)
            #print(output[0][0], output[1][0], output[2][0], output[3][0], value)
            
            if float(value) > 0.5:
                plt.plot(point[0], point[1],'yo')

    # draw the train input points.
    inputf = open('two_spiral_train.txt')
    lines = inputf.readlines()
    for line in lines:
        words = line.split()
        if float(words[2]) > 0.5:
            plt.plot(float(words[0]), float(words[1]),'ko')
        else:
            plt.plot(float(words[0]), float(words[1]),'ro')


    # draw the sub-problem classification boundary.
    for k in range(4):
        plt.figure(k+2)
        plt.axis([-size,size, -size,size])
        ax = plt.gca()
        ax.set_aspect(1)

        title = "learning_rate=%.2f, e_threshold=%.2f" %(MLQPs[0].alpha, MLQPs[0].e_threshold)
        plt.title(title)
        xlabel = "X axis (sub-problem %d, iterate_count=%d)" % (k, MLQPs[k].cnt_iterate)
        plt.xlabel(xlabel)
        plt.ylabel('Y axis')

        print("Painting No.%d plot..." % k)

        # draw the classification boundary
        for i in range(grain*2):
            for j in range(grain*2):
                point = [(grain - float(i))/grain*size, (grain - float(j))/grain*size]
        
                MLQPs[k].add_case(point, [0])
                if MLQPs[k].forward()[0] > 0.5:
                    plt.plot(point[0], point[1],'yo')

        # draw the input points
        for line in train_sets[k]:
            words = line.split()
            if float(words[2]) > 0.5:
                plt.plot(float(words[0]), float(words[1]),'ko')
            else:
                plt.plot(float(words[0]), float(words[1]),'ro')
            
    inputf.close()

    # display the final plot.
    plt.show()
    
    


