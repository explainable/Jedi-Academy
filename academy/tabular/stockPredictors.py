#Program to create classifiers and regressors on a variety of datasets:
#(male/female) voice data, boston housing dataset, titanic dataset

from models import *
from deciViz import plotTree, convertDotData, convertTreeToParentChild, getNode
import sys
import numpy as np
import pickle

__dir__ = "/".join(__file__.rsplit('/')[:-1])
#Read in titanic.csv and return a 2D list of instances, 1D list of labels, and
#1D list of feature names
def readInTitanic():
    allInsts = []
    allLabels = []
    featNames = []
    with open(__dir__ + "data/titanic.csv") as titFile:
        for linenum, line in enumerate(titFile):
            features = line.split(",")
            currInst = []
            for index, feat in enumerate(features):
                if linenum == 0:
                    if index != 0 and index != 2:
                        featNames.append(feat.replace('\r', '').replace('\n',''))
                else: 
                    if index == 0:
                        allLabels.append(int(feat))
                    elif index != 2:
                        if feat == "male":
                            currInst.append(0.0)
                        elif feat == "female":
                            currInst.append(1.0)
                        else:
                            currInst.append(float(feat))
            if linenum != 0:
                allInsts.append(currInst)

    return allInsts, allLabels, featNames

#Accept the max_depth of the tree. Create a decision tree to classify voice
#data
#
#Return dot_data, a string describing the nodes and the edges in the resulting
#tree
def treeVoice(maxDepth=100):
    voicefeat = [ [180, 15,0],     
              [177, 42,0],
              [136, 35,1],
              [174, 65,0],
              [141, 28,1]]

    voiceLabel = ['man', 'woman', 'woman', 'man', 'woman']    

    data_feature_names = [ 'height', 'hair length', 'voice pitch' ]

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
    voicefeat = [ [180, 15,0],     
              [177, 42,0],
              [136, 35,1],
              [174, 65,0],
              [141, 28,1]]

    voiceLabel = ['man', 'woman', 'woman', 'man', 'woman']    
    voiceLabel = [0, 1, 1, 0, 1] 

    clf, params, predInsts = linClassify(voicefeat, voiceLabel, voicefeat)
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
    pickle.dump(clf, open("bostonTree.pkl", 'wb'))
    #convertTreeToParentChild(clf, featNames, predLabels, deciPath)

    #plotTree(bostonData, "bostonReg.png")

    features, labels, featNames = readInTitanic()
    titDepth = 5
    clf, featNames, predLabels, deciPath = treeTitanic(titDepth)
    convertTreeToParentChild(clf, featNames, predLabels, deciPath)
    print(getNode([1, 0, 50, 0, 0, 152], clf, deciPath, featNames))
    pickle.dump(clf, open("titanicTree.pkl", 'wb'))

    #plotTree(titData, "titanicClass.png")

    #params, intercept, labels = linBoston()
    #for index, feat in enumerate(params):
        #print "feat %d, %5.3f"  % (index, feat)
    #params, intercept, labels = linVoice()
    #params, intercept, labels = linTitanic()
    #print params, labels

testMain()
