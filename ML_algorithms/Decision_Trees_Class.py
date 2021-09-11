#  /******************************************************************************
#   * Copyright (c) - 2021 - Anna Chechulina                                  *
#   * The code in Decision_Trees_Class.py  is proprietary and confidential.                  *
#   * Unauthorized copying of the file and any parts of it                       *
#   * as well as the project itself is strictly prohibited.                      *
#   * Written by Anna Chechulina  <chechulinaan17@gmail.com>,   2021                 *
#   ******************************************************************************/

import numpy as np

class DecisionTreeID3():
    def __init__(self, target_name='recommendation'):
        """
        tsrget_name - name for target column
        first_gini - list for counting gini
        """
        self.target_name=target_name
        self.first_gini=[] 
        self.result=0
        
    def _gini(self, data, feature_name,category,target_name):
        """
        Description:
        Function for calculating gini impurity(G_i = 1- sum(P_i_k)^2).

        Args:
        data - dataset of data(pandas.DataFrame)
        feature_name - list of features 
        category - list of categories (what can be inside every feature)
        target_name - target name. 

        Returns:
        gini-impurity 
        """
        #calculate the length for category in features
        common_length=len(data[data[feature_name]==category])    
        def P_i_K(target, feature_name):
            #calculate probability for every variants in target column for avary category in feature
            return len(data[(data[feature_name]==category) & (data[target_name]==target)])/common_length
        #impurity of every category in feature 
        gini_impurity=1-sum(P_i_K(target, feature_name)**2 for target in set(data[target_name])) 
        return gini_impurity


    def _total_gini(self, data, feature_name, target_name):
        """
        Description:
        Function for calculation total gini impurity (G=sum(Gini_i*P_k_a))

        Args:
        data - dataset of data(pandas.DataFrame)
        feature_name - list of features 
        target_name - target name. 

        Returns:
        Total gini  - information gain
        """
        def P_k_a(category, feature_name):
            #probability for every category in whole data
            return len(data[data[feature_name]==category])/len(data)
        #for every category in feature
        for category in set(data[feature_name]):
            #calculate information gain for every feature
            gini_value=self._gini(data, feature_name, category, self.target_name)
            P_k_a_value=P_k_a(category, feature_name)
            self.result+=gini_value*P_k_a_value
        return self.result
  

    def _result(self, data):
        """
        Description:
        Function for collecting information gains for all features for every node


        Args:
        data - dataset of data(pandas.DataFrame)

        Returns:
        First_gini - list of result for total gini impurity
        """
        #feature names - all feature before last column
        feature_names=data.keys()[:-1]
        for feature_name in feature_names: 
            self.first_gini.append(self._total_gini(data, feature_name, self.target_name))
        return self.first_gini
    
    
    def _buildTree(self, data, tree=None):
        """
        Description:
        Function for building tree
        
        Args: 
        data - dataset of data(pandas.DataFrame)
        tree  - vocabulary with tree
        
        Returns:
        tree"""
        #features which we will use for predictions, future nodes
        feature_names=data.keys()[:-1]
        #dictionary with features+information gain for them(can be via entropy or gini)
        voc=dict(zip(feature_names,self._result(data)))
        
        #if we use gini then we will choose minimum one
        node=min(voc, key=voc.get)
        
        if tree is None:
            tree={}
            tree[node]={}

        attributes=np.unique(data[node])

        #just go deep into the tree, for every nodes we cheking information gain 
        for value in attributes:
            df_new_2=data[data[node]==value]
            clValue,counts=np.unique(df_new_2[self.target_name], return_counts=True)

            if len(counts)==1:
                tree[node][value]=clValue[0]
            else:
                tree[node][value]=self._buildTree(df_new_2.drop(columns=node))
        
        return tree
    
    
    def _predict(self, inst, tree):
        """
        Description:
        Function for prediction using tree

        Args:
        inst- data for prediction - pandas.Series
        tree- tree from training
        
        Returns:
        prediction
        """
        for nodes in tree.keys():
            value=inst[nodes]
            tree=tree[nodes][value]
            prediction=0
            
            if type(tree) is dict:
                prediction=self._predict(inst,tree)
            else:
                prediction=tree
                break;

        return prediction