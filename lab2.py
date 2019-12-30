__author__ = "Milind Kamath mk6715"

import re
import math
import pickle
import sys


#feature matrix
featurematrix = ["Double quotes", "Character Names", "Specific Dialogues?", "MysteryWords", "Poem?",
                 "Certain characters", "Archaic english words?", "Arthur(1)/Herman(0)"]


# tree node
class node:

    __slots__ = "featureName", "eLeft", "eRight", "leftlink", "rightlink"

    def __init__(self, feature, left, right, yes=None, no=None):
        self.featureName = feature
        self.eLeft = left
        self.eRight = right
        self.leftlink = yes
        self.rightlink = no

    def __str__(self):
        return self.featureName + " " + str(self.eLeft) + " " + str(self.eRight)


# feature #1
def doublequotes(para):
    flag = False
    lines = para.split("\n")
    for line in lines:
        if re.search('\".*"', line):
            flag = True
            break
        else:
            flag = False
    if flag:
        return 1
    else:
        return 0


# feature # 2
def characters(para):
    flag = False
    lines = para.split("\n")
    for line in lines:
        if re.search("\s*[w,W]atson\s*", line) or re.search("\s*[h,H]olmes\s*", line) or \
                re.search("\s*[i,I]rene\s*", line) or re.search("\s*[a,A]dler\s*", line) or \
                re.search("\s*[j,J]ohn\s*", line) or re.search("\s*[s,S]herlock\s*", line) or \
                re.search("\s*221[b,B]\s*", line) or re.search("\s*[b,B]aker\s[s,S]treet\s*", line) or \
                re.search("\s*[h,H]udson\s*", line) or re.search("\s*[m,M]ycroft\s*", line) or \
                re.search("\s*[m,M]oriarty\s*", line) or re.search("\s*[e,E]ngland\s*", line) or\
                re.search("\s*[l,L]estrarde\s*", line) or re.search("\s*[m,M]ary\s*", line):
            flag = True
            break
        else:
            flag = False
    if flag:
        return 1
    else:
        return 0


# feature # 3
def dialogues(para):
    flag = False
    lines = para.split("\n")
    for line in lines:
        if re.search("\s*[m,M]y\sdear\s[w,W]atson\s*", line) or re.search("\s*[e,E]ast\swind\s*", line):
            flag = True
            break
        else:
            flag = False
    if flag:
        return 1
    else:
        return 0


# feature # 4
def mysteryWords(para):
    flag = False
    lines = para.split("\n")
    for line in lines:
        if re.search("\s*[g,G]hosts*\s*", line) or re.search("\s*[s,S]ecrets*\s*", line) or \
                re.search("\s*[d,D]eaths*\s*", line) or re.search("\s*[m,M]ystery\s*", line) or \
                re.search("\s*[c,C]onfess(ed)\s*", line) or re.search("\s*[d,D]anger\s*", line) or \
                re.search("\s*[b,B]lood\s*", line) or re.search("\s*[d,D]educ[e,tion]\s*", line):
            flag = True
            break
        else:
            flag = False
    if flag:
        return 1
    else:
        return 0


# feature # 5
def isPoem(para):
    sum = 0
    lines = para.split("\n")
    for line in lines:
        sum += len(line)
    if sum/len(lines) < 40:
        return 1
    else:
        return 0


# feature # 6
def certaincharacters(para):
    flag = False
    lines = para.split("\n")
    for line in lines:
        if re.search('.*-.*', line) or re.search('.*--.*', line):
            flag = True
            break
        else:
            flag = False
    if flag:
        return 1
    else:
        return 0


# feature # 7
def archaicenglishwords(para):
    flag = False
    lines = para.split("\n")
    for line in lines:
        if re.search('\s*[n,N]igh\s*', line) or re.search('\s*[a,A]+fore\s*', line) or re.search('\s*[e,E]ft\s*', line) or \
            re.search('\s*[f,F]ore\s*', line) or re.search('\s*[h,H]ither\s*', line) or re.search('\s*[t,T]hereinto\s*', line) \
                or re.search('\s*[t,T]hy\s*', line) or re.search('\s*[b,B]etwixt\s*', line) or re.search('\s*[t,T]hou\s*', line):
            flag = True
            break
        else:
            flag = False
    if flag:
        return 1
    else:
        return 0


# find all features in a paragraph
def feature(para):
    val = []
    bool = doublequotes(para)
    val.append(bool)
    bool = characters(para)
    val.append(bool)
    bool = dialogues(para)
    val.append(bool)
    bool = mysteryWords(para)
    val.append(bool)
    bool = isPoem(para)
    val.append(bool)
    bool = certaincharacters(para)
    val.append(bool)
    bool = archaicenglishwords(para)
    val.append(bool)
    return val


# dot product
def dotProduct(vec1, vec2):
    dot = 0
    for v1, v2 in zip(vec1, vec2):
        dot += v1 * v2
    return dot


# sigmoid function
def sigmoid(value):
    den = 1 + math.exp(-value)
    sig = 1 / den
    return sig


# learn weights
def adjustWeight(alpha, vecW, vecV, sig):
    y = vecV[-1:][0]
    newvecW = []
    for w, v in zip(vecW, vecV):
        newvecW.append(w + alpha * (y - sig) * sig * (1 - sig) * v)
    return newvecW


# learning single layer perceptron
def learn(valueVector, alpha, weightVector, threshold):
    for i in range(threshold):
        val = dotProduct(valueVector[i][:-1], weightVector)
        sig = sigmoid(val)
        weightVector = adjustWeight(alpha, weightVector, valueVector[i], sig)
        alpha = alpha / 4
    return weightVector


# predicting using single layer perceptron
def predict(valV, wtV):
    val = dotProduct(valV, wtV)
    value = sigmoid(val)
    if value > 0.5:
        return 1
    else:
        return 0


# calculate gain
def gain(Einit, Eleft, Eright, pos, neg, tot):
    Gain = round(Einit - ((round((pos / tot), 2) * Eleft + round((neg / tot),2) * Eright)), 2)
    return Gain


# calculate entropy
def Ent(pos, neg, tot):
    if tot != 0:
        probpos = round((pos / tot), 2)
        probneg = round((neg / tot), 2)
    else:
        probpos = pos
        probneg = neg
    if probpos == 0 or probneg == 0:
        return 0
    else:
        E = round(- (probpos * math.log(probpos, 2)) - (probneg * math.log(probneg, 2)), 2)
    return E


# modify matrix for next recursion
def changeMatrix(matrix, index, decision):
    del featurematrix[index]
    newmatrix = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])-1):
            if j == index:
                if matrix[i][j] == decision:
                    newmatrix.append(matrix[i][:index] + matrix[i][index+1:])
                    break
    if newmatrix == []:
        return matrix[0][-1]
    else:
        return newmatrix


# Decision tree code
def DTree(matrix, Entropy=1):
    if len(featurematrix) - 1 == 0:
        return
    if Entropy == 0:
        return matrix
    Node, Eleft, Eright = bestAttribute(matrix)
    temp = node(featurematrix[Node], Eleft, Eright)
    forleftnewmatrix = matrix[:]
    temp.leftlink = DTree(changeMatrix(forleftnewmatrix, Node, 1), Eleft)
    featurematrix.insert(Node, temp.featureName)
    forrightnewmatrix = matrix[:]
    temp.rightlink = DTree(changeMatrix(forrightnewmatrix, Node, 0), Eright)
    featurematrix.insert(Node, temp.featureName)
    return temp


# find bestattribute
def bestAttribute(matrix):
    attr = 0
    ELEFT = 0
    ERIGHT = 0
    maxGain = -1
    countYes = 0
    countNo = 0
    EleftYes = 0
    ELeftNo = 0
    ErightYes = 0
    ErightNo = 0
    EntLeft = 0
    EntRight = 0
    for j in range(len(matrix[0])-1):
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                countYes += 1
            else:
                countNo += 1
            if matrix[i][j] == 1 and matrix[i][len(matrix[0])-1] == 1:
                EleftYes += 1
            elif matrix[i][j] == 1 and matrix[i][len(matrix[0])-1] == 0:
                ELeftNo += 1
            elif matrix[i][j] == 0 and matrix[i][len(matrix[0])-1] == 1:
                ErightYes += 1
            elif matrix[i][j] == 0 and matrix[i][len(matrix[0])-1] == 0:
                ErightNo += 1
        Einit = Ent(countYes, countNo, countYes + countNo)
        EntLeft = Ent(EleftYes, ELeftNo, EleftYes + ELeftNo)
        EntRight = Ent(ErightYes, ErightNo, ErightYes + ErightNo)
        tempGain = gain(Einit, EntLeft, EntRight, EleftYes + ELeftNo, ErightYes + ErightNo, len(matrix))
        if tempGain > maxGain:
            maxGain = tempGain
            attr = j
            ELEFT = EntLeft
            ERIGHT = EntRight
            YES = countYes
            NO = countNo
        countYes = 0
        countNo = 0
        EleftYes = 0
        ELeftNo = 0
        ErightYes = 0
        ErightNo = 0
    return attr, ELEFT, ERIGHT


# main
def main():
    weights = []
    stepAlpha = 0.8
    matrix = []
    action = ""
    predictfilename = ""
    if len(sys.argv) == 2:
        action = sys.argv[1]
    else:
        action = sys.argv[1]
        predictfilename = sys.argv[2]
    if action == 'train' or action == 'Train' or action == 'TRAIN':
        data = open('train.txt', 'r', encoding='utf8').read().split('\n\n')
        for para in data:
            x = para.rfind('$')
            paragraph = para[:x]
            author = para[x+1:]
            val = feature(paragraph)
            if author == "A":
                val.append(1)
            elif author == "H":
                val.append(0)
            matrix.append(val)

        # Single layer perceptron training
        print("Training through Single Layer Perceptron")
        threshold = len(matrix)
        for i in range(len(matrix[0])-1):
            weights.append(0)
        weights = learn(matrix, stepAlpha, weights, threshold)
        wt = open('SlptrainedData.txt', 'w')
        for w in weights:
            wt.write(str(w) + "\n")
        print("Single layer perceptron trained")

        # decision tree training
        print("Training through Decision Trees")
        tree = DTree(matrix)
        treeinst = open('state.txt', 'wb')
        pickle.dump(tree, treeinst)
        treeinst.close()
        print("Decision tree trained")
    else:
        # single layer perceptron prediction
        print("Predicting using single layer perceptron")
        adjustedWeights = []
        with open('SlptrainedData.txt', 'r') as w:
            val = w.read().split("\n")
            for i in range(len(val)-1):
                adjustedWeights.append(float(val[i]))

        print("Predicting from a paragraph in predict.txt file")
        f = open(predictfilename, 'r')
        para = f.read()
        valVector = feature(para)
        authorclass = predict(valVector, adjustedWeights)
        if authorclass == 0:
            print("Written by Herman Melville")
        else:
            print("Written by Arthur Conan Doyle")

        # decision tree prediction
        instance = open('state.txt', 'rb')
        tree = pickle.load(instance)
        instance.close()


# start
if __name__ == '__main__':
    main()
