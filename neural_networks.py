import numpy as np
x=np.array(([2,9],[1,5],[3,6]),dtype='float')
y=np.array(([92],[86],[89]),dtype='float')
x=x/np.amax(x,axis=0)
y=y/100

class neural:
    def __init__(self):
        #parameters
        self.inputsize=2
        self.outputsize=1
        self.hiddensize=3
        self.w1=np.random.randn(self.inputsize,self.hiddensize) ##weight matrix between Layer1 and Layer2 (3x2)
        self.w2=np.random.randn(self.hiddensize,self.outputsize)##weight matrix between Layer2 and Layer3)(3x1)
        
    def feedforward(self,x):
        #forward propagation
        self.z=np.dot(x,self.w1) #dot product of Input matrix(x) and weight matrix between layer1 and Layer2(hidden Layer)
        self.z1=self.sigmoid(self.z)  #scaling down the dot product using sigmoid function
        self.z2=np.dot(self.z1,self.w2)  #dot product of z1 and weigth matrix between Layer2(hidden layer) and Layer3
        output=self.sigmoid(self.z2)
        return output
    # sigmoid function for scaling down values to the range of 0-1
    def sigmoid(self,s,deriv=False):
        if(deriv==True):
            return s*(1-s)
        return 1/(1+np.exp(-s))

    def backprop(self,x,y,output):
        self.output_error=y-output #error in output
        self.output_delta=self.output_error* self.sigmoid(output,deriv=True)

        self.z1_error=self.output_delta.dot(self.w2.T) #z1 error- how much our hidden layer weights contribute to the output error
        self.z1_delta=self.z1_error*self.sigmoid(self.z1,deriv=True)

        self.w1+=x.T.dot(self.z1_delta)
        self.w2+=self.z1.T.dot(self.output_delta)

    def train(self,x,y):
        output=self.feedforward(x)
        self.backprop(x,y,output)

NN=neural()
for i in range (1000):
    NN.train(x,y)

print(str(NN.feedforward(x)))
print(str(y))
            
    