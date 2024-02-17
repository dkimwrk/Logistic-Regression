import numpy as np
import argparse
import math


def sigmoid1(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def sigmoid(x):
    
    return 1 / (1 + math.e**(-x))


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    pass


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    pass


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    count=0
    for i in range(y_pred.shape[0]):
        if y_pred[i]!=y[i]: count+=1
    return count/y_pred.shape[0]


def format_to_six_decimals(number):
    return '{:.6f}'.format(number)


if __name__ == '__main__':
       parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    train= np.loadtxt(args.train_input, delimiter='\t')
    '''print(train.shape)
    print(type(train))'''

    Xtrain= train[:, 1:]
    Xtrain =  np.c_[np.ones((Xtrain.shape[0], 1)), Xtrain]
    Ytrain= train[:, 0]   
    weights= np.zeros(Xtrain.shape[1])
    learningrate=format_to_six_decimals(args.learning_rate)
    learningrate= float(learningrate)
    '''print("learningrate")
    print(format_to_six_decimals(learningrate))
    print(type(learningrate))
    print(Xtrain[:,0])'''
    for epoch in range(int(args.num_epoch)):
        for i in range(Xtrain.shape[0]):
            gradient =  Xtrain[i]* (sigmoid(np.dot(Xtrain[i], weights)) - Ytrain[i])
            weights = weights - (learningrate* gradient)
            #print("gradient")
            #print(gradient)
            #print("weights")
            #print(weights)

    bias = weights[0]
    weights = weights[1:]
    '''print(bias)
    print(weights)'''
    '''print("bias")
    print(bias)
    print("weights")
    print(weights)'''

    trainpredictions = np.array([])
    for i in range(Xtrain.shape[0]):
        pred = np.dot(Xtrain[i][1:],weights) + bias
        guess = sigmoid(pred)
        if guess>=0.5: trainpredictions= np.append(trainpredictions, 1)
        else: trainpredictions= np.append(trainpredictions, 0)

    '''print(trainpredictions)
    print(compute_error(trainpredictions, Ytrain))
    '''
    test= np.loadtxt(args.test_input, delimiter='\t')
    Xtest= test[:, 1:]
    Ytest= test[:, 0] 
    '''print(Xtest.shape)
    print(Ytest.shape)
    print(weights.shape)
    print(bias)'''

    testpredictions = np.array([])
    for i in range(Xtest.shape[0]):
        pred = np.dot(Xtest[i],weights) + bias
        guess = sigmoid(pred)
        if guess>=0.5: testpredictions= np.append(testpredictions, 1)
        else: testpredictions= np.append(testpredictions, 0)
    
    '''print(testpredictions)
    print(compute_error(testpredictions, Ytest))'''

    predicted_train = open(args.train_out, 'w')
    for i in trainpredictions:
        predicted_train.write(str(i))
        predicted_train.write("\n")
    predicted_train.close()

    predicted_test = open(args.test_out, 'w')
    for i in testpredictions:
        predicted_test.write(str(i))
        predicted_test.write("\n")
    predicted_test.close()

    met = open(args.metrics_out, 'w')
    met.write("error(train): ")
    met.write(str(compute_error(trainpredictions, Ytrain)))
    met.write("\n")
    met.write("error(test): ")
    met.write(str(compute_error(testpredictions, Ytest)))
    met.write("\n")
    met.close()








