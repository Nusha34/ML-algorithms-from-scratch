#  /******************************************************************************
#   * Copyright (c) - 2021 - Anna Chechulina                                  *
#   * The code in Naive_Bayes_Class.py  is proprietary and confidential.                  *
#   * Unauthorized copying of the file and any parts of it                       *
#   * as well as the project itself is strictly prohibited.                      *
#   * Written by Anna Chechulina  <chechulinaan17@gmail.com>,   2021                 *
#   ******************************************************************************/

import os
import re
import codecs
import math

class NaiveBayesClassifier:
"""
Naive Bayes Classifier
Input - folder with classes and files inside classes
Output - which class does the documents belong to
"""
    
    def __init__(self):
        """
        vocabulary - the vocabulary with with words and their number
        num_docs - the number of docs
        classes - classes
        priors - the prior probabilities
        conditionals - conditionals
        """

        self.vocabulary = {}
        self.num_docs = 0
        self.classes = {}
        self.priors = {}
        self.conditionals = {}

    
    def _tokenize_str(self, doc):
        """
        Function for tokenize string
        """
        return re.findall(r'\b\w\w+\b', doc) # return all words with #characters > 1

    
    def _tokenize_file(self, doc_file):#reading document, encoding and tokenizing
        """
        Function for tokenize file

        Input - file
        Output - tokenizing file
        """
        with codecs.open(doc_file, encoding='latin1') as doc:
            doc = doc.read().lower()
            _header, _blankline, body = doc.partition('\n\n')
            return self._tokenize_str(body) # return all words with #characters > 1

    
    def train(self, path):
        """
        Function for train model

        Input - path where is train file
        """
        for class_name in os.listdir(path):
            self.classes[class_name] = {"doc_counts": 0, "term_counts": 0, "terms": {}}
            path_class = os.path.join(path, class_name)
            for doc_name in os.listdir(path_class):
                terms = self._tokenize_file(os.path.join(path_class, doc_name))
                self.num_docs += 1
                self.classes[class_name]["doc_counts"] += 1

                # build vocabulary and count terms
                for term in terms:
                    self.classes[class_name]["term_counts"] += 1
                    if not term in self.vocabulary:
                        self.vocabulary[term] = 1
                        self.classes[class_name]["terms"][term] = 1
                    else:
                        self.vocabulary[term] += 1
                        if not term in self.classes[class_name]["terms"]:
                            self.classes[class_name]["terms"][term] = 1
                        else:
                            self.classes[class_name]["terms"][term] += 1
                            
                            
        for cn in self.classes:
        # calculate priors
        # P(C = 11) = 600/20000 --> 0.03
        # log(P(C = 11)) = log(600)-log(20000)
        #classes[cn]['doc_counts'] - the number of documents inside class
        #num_docs - the number of docs inside folder
            self.priors[cn] = math.log(self.classes[cn]['doc_counts']) - math.log(self.num_docs)
            #calculate conditionals
            #classes[cn]['terms'] - the number of termins inside class - P(X|y)
            self.conditionals[cn] = {}
            #termins for one class
            cdict = self.classes[cn]['terms'] 
            #sum all termins
            c_len = sum(cdict.values())
            for term in self.vocabulary:
                t_ct = 1.
                t_ct += cdict[term] if term in cdict else 0.
                #if you've word then we're trying to find probabillity that this word attend to class
                self.conditionals[cn][term] = math.log(t_ct) - math.log(c_len + len(self.vocabulary))
        
        
    
    def test(self, path_test):
        """
        Function for test model

        Input - path where is test file
        Output - predictions
        """
        predictions=[]
        print("Testing <%s>" % path_test)
        for class_num, class_name in enumerate(self.classes):
            for doc in os.listdir(os.path.join(path_test, class_name)):
                doc_path=os.path.join(path_test, class_name, doc)
                #print(doc_path)
                token_list = self._tokenize_file(doc_path)
                result = self._scores(doc, token_list)
                predictions.append(result)        
        return predictions
    
    
    def _scores(self, doc, tokens):
        """
        Function for count scores

        Input - document and tokenize file
        Output - scores
        """
        scores = {}
        scores[doc] = {}
        for class_num, class_name in enumerate(self.classes):
            scores[doc][class_name] = self.priors[class_name]
            for term in tokens:
                    if term in self.vocabulary:
                        scores[doc][class_name] += self.conditionals[class_name][term]
        return scores
    
    
    def predict(self, test_doc_path):
        """
        Function for prediction

        Input - path where is test file
        Output - result of scores function
        """

        doc= test_doc_path.split('\\')[-1]
        token_list_predict = self._tokenize_file(test_doc_path)
        return self._scores(doc, token_list_predict)