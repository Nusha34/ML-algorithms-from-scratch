#  /******************************************************************************
#   * Copyright (c) - 2021 - Anna Chechulina                                  *
#   * The code in Linear_Regression_Class.py  is proprietary and confidential.                  *
#   * Unauthorized copying of the file and any parts of it                       *
#   * as well as the project itself is strictly prohibited.                      *
#   * Written by Anna Chechulina  <chechulinaan17@gmail.com>,   2021                 *
#   ******************************************************************************/

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression model

    Input - variables for training and prediction
    Output - predicted depended variables
    """
        
    def __init__(self, L=0.001, epochs=100):
        """
        L - lambda for new weights
        epochs - number epochs for gradient descent
        """
        self.L=L
        self.epochs=epochs
    
    def get_weight(self, values, pred_values):
        """
        Function for getting weights
        tX-transpose matrix
        inv_matrix - inverse matrix

        Input - dependent and independent variables
        Output - weights
        """
        tX=values.transpose()
        inv_matrix=np.linalg.inv(tX.dot(values))
        weight=inv_matrix.dot(tX).dot(pred_values).transpose()[0]
        return weight


    def predict(self, values, weight):
        """
        Function for getting new dependent value

        Input  - independent variables and new weights
        Output - prediction for y(dependent variable)
        """
        predict_all=[]
        for i in range(len(values)):
            predict= sum(values[i]*weight)
            predict_all.append([predict])    
        return predict_all
       
    def normalization(self, ind_values):
        """
        Function for normalizing our data

        Input - independent variables
        Output - normalization values
        """
        scaler = MinMaxScaler().fit(ind_values)
        ind_values= scaler.transform(ind_values)
        return ind_values
    
    def accuracy(self, predicted_dep_values, dep_values):
        """
        Function for calculating accuracy
        """
        print('---Accuracy(MSE)---\n', (np.square(predicted_dep_values - dep_values)).mean())
        
    
    def train(self, ind_values, dep_values):
        """
        Function for training
        ind_values - x values after normalization 
        weight - weight from which we start train our model 
        y_pred - first prediction for y
        mse_list - counting mse with every iteration
        D - gradient
        L - lambda which we put by hand


        Output - new weights
        """
        ind_values=self.normalization(ind_values)
        weight=self.get_weight(ind_values, dep_values)
        y_pred=self.predict(ind_values, weight)
        mse_list_for_graphs=[]
        for i in range(self.epochs):
            y_pred=self.predict(ind_values,weight)
            D = np.array([(-2*(ind_values[:,j].T.dot(dep_values-y_pred))).mean() for j in range(len(ind_values[0]))])
            weight=weight-self.L*D
            y_pred_new=self.predict(ind_values,weight)
            mse=(np.square(dep_values - y_pred_new)).mean()
            mse_list_for_graphs.append(mse)
        print('--L--\n', self.L)
        print('--epochs--\n', self.epochs)
        print('--Graph for mse--\n')    
        plt.plot(mse_list_for_graphs)
        return weight 