#Program to create classifiers and regressors on a variety of datasets:
#(male/female) voice data, boston housing dataset, titanic dataset

from models import *
from deciViz import plotTree

#Read in titanic.csv and return a 2D list of instances, 1D list of labels, and
#1D list of feature names
def readInTitanic():
    allInsts = []
    allLabels = []
    featNames = []
    with open("titanic.csv") as titFile:
        for linenum, line in enumerate(titFile):
            features = line.split(",")
            currInst = []
            for index, feat in enumerate(features):
                if linenum == 0:
                    if index != 0 and index != 2:
                        featNames.append(feat)
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
    return dot_data

#Perform a linear regression on boston housing data
#
#Return the resulting weights of the model and the predicted training instances
def linBoston():
    features, labels = load_boston(True)
    clf, params, predLabels = linRegress(features, labels, features)
    return params, predLabels

#Accept the max_depth of the tree. Create a decision tree to perform a
#classification on the titanic dataset
#
#Return dot_data, a string describing the nodes and the edges in the resulting
#tree
def treeTitanic(maxDepth=100):
    features, labels, featNames = readInTitanic()
    clf, dot_data, predLabels = treeClassify(features, labels, featNames, maxDepth, features)

#Perform a logistic regression on boston housing data
#
#Return the resulting weights of the model and the predicted training instances
def linTitanic():
    features, labels, featNames = readInTitanic()
    clf, params, predLabels = linClassify(features, labels, features)
    return params, predLabels

#self explanatory
def testMain():
    voiceDepth = 5
    voiceData = treeVoice(voiceDepth)
    plotTree(voiceData, "voiceClass.png")

    bostonDepth = 100
    bostonData = treeBoston(bostonDepth)
    plotTree(bostonData, "bostonReg.png")

    titDepth = 100
    titData = treeTitanic(titDepth)
    plotTree(titData, "titanicClass.png")

    params, labels = linBoston()
    params, labels = linVoice()
    params, labels = linTitanic()
    #print params, labels

testMain()
