#Program used to visualize decision trees. Outputs pngs

#import pydotplus
#import collections
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#Takes dot_data, a string describing the nodes and edges of a given decision
#tree. This string is returned by a treeReg or treeClassify model in the
#model.py file. Additionally, it takes filename, the name of the file where the
#image will be saved
#
#This function returns a graphical representation of a decision tree in the
#form of a png
def plotTree(dot_data, filename):
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(filename)

#Take a decision tree classifier and return the various components:
# number of nodes, left children, right children, features, and thresholds
def obtainTreeInfo(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    return n_nodes, children_left, children_right, feature, threshold

#Take a decision tree classifier
#
#Return a list with each entry being a boolean that corresponds to whether that
#node index is a leave node or not
def obtainLeaves(clf):
    n_nodes,children_left,children_right,feature,threshold = obtainTreeInfo(clf)

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return is_leaves

"""
#Take an instance of values, a decision tree classifer, a decision path matrix,
#and a list of feature names
#
#Return the leaf that the instance will be computed to. This will be in the
#form of, for example, Age<23
def getNode(instance, clf, deciPath, featNames):
    n_nodes,children_left,children_right,feature,threshold = obtainTreeInfo(clf)

    instance = np.array(instance).reshape(1,-1)
    node_indicator = clf.decision_path(instance)
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

    leave_id = clf.apply(instance)

    print(node_index)
    

    nodeVal = "-1"
    for node_id in node_index:
        if (instance[0][feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        nodeVal = "%d: %s%s%f" % (node_id, featNames[feature[node_id]],\
                threshold_sign, threshold[node_id])
        print(nodeVal)

    return nodeVal

"""

def getNode(instance, clf, deciPath, featNames):
    n_nodes,children_left,children_right,feature,threshold = obtainTreeInfo(clf)
    
    currNode = 0

    threshold_sign = "-1"
    currFeat = "-6"
    currThresh = "-7"
    i = 0

    is_leaves = obtainLeaves(clf)

    while i < 10:
        i += 1
        currFeat = feature[currNode]
        currThresh = threshold[currNode]
        if instance[feature[currNode]] <= threshold[currNode]:
            currNode = children_left[currNode]
            threshold_sign = "<="
            if is_leaves[currNode]:
                break
        else:
            currNode = children_right[currNode]
            threshold_sign = ">"
            if is_leaves[currNode]:
                break

    nodeVal = "%d: %s%s%f" % (currNode, featNames[currFeat],\
                              threshold_sign, currThresh)
    return nodeVal


#Takes a decision tree classifier, a list of feature names, a list of
#predicition labels, and a decision_path object
#
#This function will take a classifier and parse it to obtain the various nodes
#in the graph, as well as the various confidence scores, and feature split of
#each of those nodes
def convertTreeToParentChild(clf, featNames, predLabels, deciPath):
    n_nodes,children_left,children_right,feature,threshold = obtainTreeInfo(clf)

    parents = [""]
    labels = ["0: Everyone"]

    numLive = 0
    numDead = 0
    for pred in predLabels:
        if pred == 1:
            numLive += 1
        if pred == 0:
            numDead += 1
    #initVal = "0: Dead = %d%%, Live = %d%%" % \
    #                            (numDead * 100 / float(numDead + numLive), \
    #                            (numLive * 100 / float(numDead + numLive)))
    initVal = numLive / float(numDead + numLive)

    vals = [initVal]

    is_leaves = obtainLeaves(clf)

    leafId = 0
    nodeVals = {0: ""}
    for i in range(n_nodes):
        if not is_leaves[i]:
            addChild(clf, i, children_left, parents, vals, labels, \
                            nodeVals, deciPath, featNames, predLabels, "<=")

            addChild(clf, i, children_right, parents, vals, labels, \
                            nodeVals, deciPath, featNames, predLabels, ">")
            
    #print(parents)
    #print(labels)
    #print(vals)

#Add a the children of node i to the parents, vals, and labels lists. Takes the
#variables with corresponding names seen in convertTreeToParentChild
def addChild(clf, i, children_list, parents, vals, labels, nodeVals, deciPath,\
        featNames, predLabels, threshVal):
    n_nodes,children_left,children_right,feature,threshold = obtainTreeInfo(clf)
    numLive = 0
    numDead = 0
    for index, pred in enumerate(predLabels):
        if deciPath[index][children_list[i]] == 1:
            if pred == 1:
                numLive += 1
            else:
                numDead += 1

    #val = "Dead = %d%%, Live = %d%%" % \
    #                    (numDead * 100 / float(numDead + numLive), \
    #                    (numLive * 100 / float(numDead + numLive)))
    val = numLive / float(numDead + numLive)
    #val = str(children_left[i]) + ": " + val
    label = featNames[feature[i]] + threshVal + str(threshold[i])
    label = str(children_list[i]) + ": " + label
    labels.append(label)
    vals.append(val)
    nodeVals[children_list[i]] = label
    parents.append(nodeVals[i])


#Takes a dot_data datatype and parse it to obtain the various nodes in the
#corresponding decision tree
def convertDotData(dot_data):
    graph = pydotplus.graph_from_dot_data(dot_data)

    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    semiSplit = dot_data.split(";")
    nodeSplit = []
    for entry in semiSplit:
        entrySplit = entry.split()
        valid = True
        for ent in entrySplit:
            if ent == "->":
                valid = False

        if valid:
            nodeSplit.append(entry)

    nodes = []
    for node in nodeSplit:
        nodes.append(node.split('\n')[1])

    #print(edges)
    nodeInfo = {}
    for index, node in enumerate(nodes):
        if index != 0 and index != 1 and index != len(nodes) - 1:
            thisNode = node.decode()
            front = thisNode.split("\\")[0]
            twoParts = front.split("\"")
            num = twoParts[0].split()[0]
            label = twoParts[1]
            nodeInfo[str(num)] = str(label)
            
    #print(nodeInfo)
