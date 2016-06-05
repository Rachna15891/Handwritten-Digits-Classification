import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    

    

def preprocess():
    
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('./mnist_all.mat')

    training = np.empty((0,784))
    train_label = []
    validation = np.empty((0,784))
    valid_label = []
    test_data = np.empty((0,784))
    test_label = []

    for i in range(10):
        train = np.array(mat.get('train'+str(i)))
        test = np.array(mat.get('test'+str(i))).astype(float)
        a = range(train.shape[0])
        aperm = np.random.permutation(a)
        A1 = train[aperm[0:1000],:]
        A2 = train[aperm[1000:],:]    
        training = np.vstack([training, A2])
        test_data = np.vstack([test_data, test])
        validation = np.vstack([validation, A1])
        for j in range(A1.shape[0]):
            valid_label.append(i)
        for j in range(A2.shape[0]):
            train_label.append(i)
        for j in range(test.shape[0]):
            test_label.append(i)
            
    training/=255
    validation/=255
    test_data/=255
    
    #Feature selection
    total = np.vstack([training, validation])
    a = np.all(total == total[0,:], axis=0)
    c = []
    for i in range(len(a)):
        if(a[i]):
            c.append(i)
    
    training = np.delete(training, c, axis=1)
    validation = np.delete(validation, c, axis =1)
    test_data = np.delete(test_data, c, axis=1)
    
    # creating bias row for hidden layer
    bias_row =np.ones((np.size(training,0),1))
    # concatenate bias with z column-wise
    training=np.concatenate((training,bias_row),axis=1)
    # creating bias row for hidden layer
    bias_row =np.ones((np.size(test_data,0),1))
    # concatenate bias with z column-wise
    test_data=np.concatenate((test_data,bias_row),axis=1)
    # creating bias row for hidden layer
    bias_row =np.ones((np.size(validation,0),1))
    # concatenate bias with z column-wise
    validation=np.concatenate((validation,bias_row),axis=1)
    
    return training, train_label, validation, valid_label, test_data, test_label 
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    #dimension 50000*10
    trainLabel = np.zeros((training_data.shape[0],10))
    z,predictedLabel=feedForward(training_data,w1,w2);

    #Iterate over all the training examples (50,000)   
    total_size = trainLabel.shape[0] 
    for i in range(total_size):
        trainLabel[i][training_label[i]] = 1
    labelSubtract = np.subtract(trainLabel,predictedLabel)
    labelSquare = np.square(labelSubtract)
    error = np.sum(labelSquare)/(total_size*2)
    regularizationFactor = (np.sum(np.square(w1))+ np.sum(np.square(w2)))*lambdaval/(2*total_size)
    obj_val = error+regularizationFactor    
    delta = (trainLabel - predictedLabel)*(1 - predictedLabel)*predictedLabel
    gradient_W2 = np.dot(np.transpose(delta),z) * -1
    
    gradient_W1 = np.dot(np.transpose(training_data),z*(z-1)*np.dot(delta,w2))
    gradient_W1 = np.delete(gradient_W1,n_hidden,axis=1)
    gradient_W1_Reg = (lambdaval*w1+np.transpose(gradient_W1))/total_size
    gradient_W2_Reg = (lambdaval*w2+gradient_W2)/total_size
    obj_grad = np.concatenate((gradient_W1_Reg.flatten(), gradient_W2_Reg.flatten()),0)
    return (obj_val,obj_grad)
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
   
    


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 

    z,o=feedForward(data,w1,w2);
    labels = o.argmax(axis=1)
    return labels

def feedForward (data,w1,w2):

    a=np.inner(data,w1)          
    z=sigmoid(a);
    # creating bias row for hidden layer
    bias_row =np.ones((np.size(z,0),1))
    # concatenate bias with z column-wise
    z=np.concatenate((z,bias_row),axis=1)
    b=np.inner(z,w2)        
    o=sigmoid(b);
    return (z,o)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]-1

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 10
				   
# set the number of nodes in output unit
n_class = 10			   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.4

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

start = time.clock()
opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

time_taken = time.clock()-start
print "Training Time (in Seconds): "+str(time_taken)
#Reshape nnParams from 1D vector into w1 and w2 matrices

w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)



print('\n Training set Accuracy: ' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
#
##find the accuracy on Validation Dataset
#
print('\n Validation set Accuracy: ' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
#
#
predicted_label = nnPredict(w1,w2,test_data)
#
##find the accuracy on Validation Dataset
#
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

#Dumping to Pickle file
#parameters = {"n_hidden": n_hidden, "w1": w1, "w2": w2, "lambda": lambdaval}
#pickle.dump(parameters, open( "params.pickle", "wb" ))
