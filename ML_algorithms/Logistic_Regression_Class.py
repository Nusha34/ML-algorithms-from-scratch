#  /******************************************************************************
#   * Copyright (c) - 2021 - Anna Chechulina                                  *
#   * The code in Logistic_Regression_Class.py  is proprietary and confidential.                  *
#   * Unauthorized copying of the file and any parts of it                       *
#   * as well as the project itself is strictly prohibited.                      *
#   * Written by Anna Chechulina  <chechulinaan17@gmail.com>,   2021                 *
#   ******************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LogisticRegression:

    def __init__(self, lr=0.01, epochs=100, theta=np.array([1, 1]), params_initial=np.array([-1, 1]), intercept=0):
        """
        lr - lambda which we chose by hand
        theta - are our parameters w of the Logistic Regression
        """
        self.lr=lr
        self.epochs=epochs
        self.theta=theta
        self.params_initial=params_initial
        self.intercept=intercept
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def compute_loss(self, X, y, intercept=0, epsilon: float = 1e-5):
        """
        Loss(w) = -(y_i*log(h_w(x_i))) + (1-y_i)*log(1-h_w(x_i))
        theta are our parameters w of the Logistic Regression

        Return loss
        """
        batch_size = len(y)
        h = self.sigmoid(np.dot(X, self.theta)+self.intercept)
        loss = (1/batch_size)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
        return loss
    
    def accuracy(self, y, y_pred):
        """
        Return accuracy score for prediction
        """
        print('---Accuracy---\n')
        return accuracy_score(y, y_pred)

    def train(self, X, Y):
        loss_list=[]
        W=self.params_initial
        for i in range(self.epochs):
            z=np.dot(X, W)
            y_pred=self.sigmoid(z)
            loss=self.compute_loss(X, y_pred, W)
            D = np.array([(-2.*np.dot(X[:,j].T, (Y-y_pred))).mean() for j in range(len(X[0]))])
            W=W-self.lr*D
            loss_list.append(loss)
        print('---epochs---\n', self.epochs)
        print('---lr---\n', self.lr)
        print('---Graph loss function---\n')    
        plt.plot(loss_list)
        self.W=W
        print('w:', self.W)
        
        
    def predict_prob(self, x_test):
        """
        Probability that data belong to one class ot to another

        eturn probability
        """
        prob=self.sigmoid(np.dot(x_test,self.W))
        return prob
    
    
    def predict(self, x_test, threshold=0.5):
        """
        Put threshold and if probability > 0.5 then it'll be class 1 if not then 0
        """
        return (self.predict_prob(x_test) >= threshold)*1