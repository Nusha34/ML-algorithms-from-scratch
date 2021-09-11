#  /******************************************************************************
#   * Copyright (c) - 2021 - Anna Chechulina                                  *
#   * The code in Neural_Network_Class_without_exercises.py  is proprietary and confidential.                  *
#   * Unauthorized copying of the file and any parts of it                       *
#   * as well as the project itself is strictly prohibited.                      *
#   * Written by Anna Chechulina  <chechulinaan17@gmail.com>,   2021                 *
#   ******************************************************************************/

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():

    def __init__(self, nn_input_dim = 2, nn_output_dim = 2, reg_lambda = 0.01):
        """
        nn_input_dim- input layer dimensionality
        nn_output_dim - output layer dimensionality
        reg_lambda - regularization strength
        """
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.reg_lambda = reg_lambda
        

    def calculate_loss(self, model, x, y):
        """
        Calculate loss after forward propagation"""
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        #a1 = self.sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        # SoftMax (output of the final layer)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(len(x)), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./len(x) * data_loss
    
    def predict(self, model, x):
        W1, b1, W2, b2= model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        #a1 = self.sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
       
    def build_model(self, x, y, nn_hdim, num_passes=20000, print_loss=False, epsilon=0.01):
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))

        # This is what we return at the end
        model = {}
        loss_list=[]
    
        # Gradient descent. For each batch...
        for i in range(0, num_passes):
                # Forward propagation
                z1 = x.dot(W1) + b1
                a1 = np.tanh(z1)
                #a1=self.sigmoid(z1)
                z2 = a1.dot(W2) + b2
                exp_scores = np.exp(z2)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Backpropagation
                delta3 = probs
                delta3[range(len(x)), y] -= 1
                dW2 = (a1.T).dot(delta3)
                db2 = np.sum(delta3, axis=0, keepdims=True)
                delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))

                dW1 = np.dot(x.T, delta2)
                db1 = np.sum(delta2, axis=0)

                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW2 += self.reg_lambda * W2
                dW1 += self.reg_lambda * W1


                # Gradient descent parameter update
                W1 += -epsilon * dW1
                b1 += -epsilon * db1
                W2 += -epsilon * dW2
                b2 += -epsilon * db2


                # Assign new parameters to the model
                model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
                # Optionally print the loss.
                # This is expensive because it uses the whole dataset, so we don't want to do it too often.
                if print_loss and i % 1000 == 0:
                    l= self.calculate_loss(model, x, y)
                    print("Loss after iteration %i: %f" %(i, l))
                    loss_list.append(l)
        print(plt.plot(loss_list))   
        return model