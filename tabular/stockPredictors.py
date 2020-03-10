#Program to create classifiers and regressors on a variety of datasets:
#(male/female) voice data, boston housing dataset, titanic dataset

from tabular.models import *
from tabular.deciViz import plotTree, convertDotData, convertTreeToParentChild, getNode
from tabular.runPickle import readInTitanic
import sys
import numpy as np

__dir__ = "/".join(__file__.rsplit('/')[:-1])

#Accept the max_depth of the tree. Create a decision tree to classify voice
#data
#
#Return dot_data, a string describing the nodes and the edges in the resulting
#tree
def treeVoice(maxDepth=100):
    voicefeat = [[180, 15, 0],
                 [177, 42, 0],
                 [136, 35, 1],
                 [174, 65, 0],
                 [141, 28, 1]]

    voiceLabel = ['man', 'woman', 'woman', 'man', 'woman']

    data_feature_names = ['height', 'hair length', 'voice pitch']

    clf, dot_data, predLabels = treeClassify(voicefeat, voiceLabel, data_feature_names, maxDepth, voicefeat)
    return dot_data

def featRanges(features):
    largestFeat = [(-1 * 1e10)] * int(features.shape[1])
    smallestFeat = [(1 * 1e10)] * int(features.shape[1])
    for inst in features:
        for index, num in enumerate(inst):
            if num < smallestFeat[index]:
                smallestFeat[index] = num
            if num > largestFeat[index]:
                largestFeat[index] = num
    featRanges = []
    for index, num in enumerate(largestFeat):
        featRanges.append((smallestFeat[index], largestFeat[index]))

    return featRanges


#Perform a logistic regression on voice data
#
#Return the resulting weights of the model and the predicted training instances
def linVoice():
    voicefeat = [[180, 15, 0],
                 [177, 42, 0],
                 [136, 35, 1],
                 [174, 65, 0],
                 [141, 28, 1]]

    voiceLabel = ['man', 'woman', 'woman', 'man', 'woman']
    voiceLabel = [0, 1, 1, 0, 1]

    clf, params, predInsts, _ = linClassify(voicefeat, voiceLabel, voicefeat)
    return params, predInsts


#Accept the max_depth of the tree. Create a decision tree to perform a regression
#on boston housing data
#
#Return dot_data, a string describing the nodes and the edges in the resulting
#tree
def treeBoston(maxDepth=100):
    boston = load_boston()
    features, labels = load_boston(True)
    data_feature_names = boston.feature_names
    clf, dot_data, predLabels = treeReg(features, labels, data_feature_names, maxDepth, features)

    boston = load_boston()
    features, labels = load_boston(True)
    featNames = boston.feature_names
    clf, predLabels, deciPath = treeReg(features, labels, featNames, maxDepth, features)
    print(predLabels)
    return clf, featNames, predLabels, deciPath

#Perform a linear regression on boston housing data
#
#Return the resulting weights of the model and the predicted training instances
def linBoston():
    features, labels = load_boston(True)
    ranges = featRanges(features)
    #print "num", len(ranges)
    #for range in ranges:
    #    print range, (range[1] - range[0])/2.0
    #print "\n\n\n\n"
    clf, params, intercept, predLabels = linRegress(features, labels, features)
    return params, intercept, predLabels

#Accept the max_depth of the tree. Create a decision tree to perform a
#classification on the titanic dataset
#
#Return dot_data, a string describing the nodes and the edges in the resulting
#tree
def treeTitanic(maxDepth=100):
    features, labels, featNames = readInTitanic()
    clf, predLabels, deciPath = treeClassify(features, labels, featNames, maxDepth, features)
    return clf, featNames, predLabels, deciPath

#Perform a logistic regression on boston housing data
#
#Return the resulting weights of the model and the predicted training instances
def linTitanic():
    features, labels, featNames = readInTitanic()
    clf, params, intercept, predLabels = linClassify(features, labels, features)
    return params, intercept, predLabels


#self explanatory
def testMain():
    #voiceDepth = 5
    #voiceData = treeVoice(voiceDepth)
    #plotTree(voiceData, "voiceClass.png")

    bostonDepth = 10
    features, labels = load_boston(True)
    clf, featNames, predLabels, deciPath = treeBoston(bostonDepth)
    #convertTreeToParentChild(clf, featNames, predLabels, deciPath)

    #plotTree(bostonData, "bostonReg.png")

    features, labels, featNames = readInTitanic()
    titDepth = 5
    clf, featNames, predLabels, deciPath = treeTitanic(titDepth)
    convertTreeToParentChild(clf, featNames, predLabels, deciPath)
    print(getNode([1, 0, 50, 0, 0, 152], clf, deciPath, featNames))

    #plotTree(titData, "titanicClass.png")

    #params, intercept, labels = linBoston()
    #for index, feat in enumerate(params):
        #print "feat %d, %5.3f"  % (index, feat)
    #params, intercept, labels = linVoice()
    #params, intercept, labels = linTitanic()
    #print params, labels

#testMain()
