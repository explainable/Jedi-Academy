#Program to return linear classifiers and regressors, as well as decision tree
#classifiers and regressors

from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree

#Perform a regression problem with a decision tree. Takes the features 
#(2D list), labels (1D list), feature names (1D list), and the maximum depth 
#of the tree being trained. A list of testing instances will also be taken 
#and the tree will run predictions on them
#
#Returns - the regressionTree itself
#        - dot_data, a string describing the nodes and edges of the tree, 
#        - predLabels, the predicted labels of the test_insts
def treeReg(features, labels, featNames, maxDepth, test_insts):
    clf = tree.DecisionTreeRegressor(max_depth = maxDepth)
    clf = clf.fit(features, labels)
    predLabels = clf.predict(test_insts)

    dot_data = tree.export_graphviz(clf,
                                feature_names=featNames,
                                out_file=None,
                                filled=True,
                                rounded=True)

    return clf, dot_data, predLabels

#Perform a regression problem with a decision tree. Takes the features 
#(2D list), labels (1D list), feature names (1D list), and the maximum depth 
#of the tree being trained. A list of testing instances will also be taken 
#and the tree will run predictions on them
#
#Returns - the classifier itself
#        - dot_data, a string describing the nodes and edges of the tree, 
#        - predLabels, the predicted labels of the test_insts
def treeClassify(features, labels, featNames, maxDepth, test_insts):
    clf = tree.DecisionTreeClassifier(max_depth = maxDepth)
    clf = clf.fit(features, labels)
    deciPath = clf.decision_path(test_insts).toarray()
    predLabels = clf.predict(test_insts)

    return clf, predLabels, deciPath

"""
    dot_data = tree.export_graphviz(clf,
                                    feature_names=featNames,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
"""


#Performs a logistic regression. Takes features (2D list), labels (1D list),
#and a list of testing instances that will be used to run predictions on 
#
#Returns - The classifier itself
#        - The coefficients of the weights of the classifier
#        - The predicted labels of test_insts
def linClassify(features, labels, test_insts):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
            multi_class='multinomial').fit(features,labels)
    predLabels = clf.predict(test_insts)
    return clf, clf.coef_, clf.intercept_, predLabels

#Performs a linear regression. Takes features (2D list), labels (1D list),
#and a list of testing instances that will be used to run predictions on 
#
#Returns - The regressor itself
#        - The coefficients of the weights of the classifier
#        - The predicted labels of test_insts
def linRegress(features, labels, test_insts):
    clf = LinearRegression()
    clf.fit(features, labels)
    predLabels = clf.predict(test_insts)
    return clf, clf.coef_, clf.intercept_, predLabels
