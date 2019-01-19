# CS 583 - Aspect Based Sentiment Analysis
# Author - Sanjay Ramachandran
# email - sramac22@uic.edu
# UIN - 671035289

from sklearn.svm import LinearSVC
from sklearn import feature_extraction, model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
from sentic import SenticPhrase
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import math
import collections
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#globals
table = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~' + "0123456789")
table2 = str.maketrans('/', ' ')
negWords = {'never','but','not','didn\'t','isn\'t','wasn\'t','doesn\'t','couldn\'t','didnt','isnt','wasnt','doesnt','couldnt'
   ,'wouldn\'t','wouldnt','don\'t','dont','can\'t','cant', 'cannot', 'wont', 'won\'t', 'havent', 'haven\'t', 'hadnt', 'hadn\'t'}
stopWords = set(stopwords.words('english')) - negWords
dataset = []
sentences = []
maxSentLen = 20
posLex = set()
negLex = set()
testSents = []
testDS = []
word2vec = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 3))

def preprocess(text, aspectTerm=None, flag=None):
    text = text.replace("[comma]", "")
    text = text.replace("  ", " ")
    text = text.replace('"', '').replace('.','').replace('(',' ').replace('(','').replace(')','').replace('!','').replace("?",'').replace("  "," ")
    text = text.lstrip('- ')
    text = text.lstrip('_')    
    text = text.lstrip('_ ')
    text = text.lstrip('-')
    text = text.rstrip(' ')
    text = text.translate(table2)
    gw = lambda w: w.lower() if w.lower() in aspectTerm else w.translate(table).lower()
    if flag:
        x = aspectTerm.replace(" ","_")
        text = text.replace(aspectTerm,x)
        temp = text.split()
        try:
            atPos = temp.index(x)
        except ValueError:
            xx = x[0].replace("_", "")
            for w in temp:
                if x in w:
                    x = w
                    atPos = temp.index(x)
                    break
                if xx in w:
                    xx = w
                    atPos = temp.index(xx)
        return [text, x+'-'+str(atPos)]
    elif aspectTerm:
        text = [gw(word) for word in text.split() 
                if len(word) > 1 and (word.lower() in aspectTerm or word.translate(table).lower() not in stopWords)]
    else:
        text = [word.lower() for word in text.translate(table2).split() if len(word) > 1]
    return text

def readTrainDataSet(file_path):
    with open(file_path) as file:
        file.readline()
        for line in file:
            data = line.split(",")
            data[2] = preprocess(data[2])
            data[1] = preprocess(data[1], data[2])
            sentences.append(preprocess(' '.join(data[1]), ' '.join(data[2]), True))
            data[-1] = data[-1].strip()
            dataset.append(data)

def readLexicons():
    with open('positive-words.txt') as file:
        for line in file:
            if len(line.strip()) > 0:
                posLex.add(line.strip())
    with open('negative-words.txt') as file:
        for line in file:
            if len(line.strip()) > 0:
                negLex.add(line.strip())

def buildTrainingData():
    X1 = []
    yForSk = []

    sp = SenticPhrase('')

    for i, data in enumerate(dataset):
        data1 = []

        # creating lexicon score with window of 5 to the left and right of aspect
        lexScore = 0
        window = 2
        atPos = int(sentences[i][1].rsplit('-', maxsplit=1)[1])
        for lind in range(atPos - window, atPos):
            if lind >= 0:
                if data[1][lind] in posLex:
                    lexScore += 1
                elif data[1][lind] in negLex or data[1][lind] in negWords:
                    lexScore -= 1
        
        for lind in range(atPos + 1, atPos + window + 1):
            if lind >= len(data[1]):
                break
            else:
                if data[1][lind] in posLex:
                    lexScore += 1
                elif data[1][lind] in negLex or data[1][lind] in negWords:
                    lexScore -= 1
        
        senScore = sp.get_polarity(sentences[i][0])        
        protoVec = np.append(Xtidf[i].A[0], [lexScore, senScore])
        
        X1.append(protoVec)
        yForSk.append(data[-1])

    X1 = np.array(X1)
    yForSk = np.array(yForSk)
    return X1, yForSk

def printScores(accuracies, precisions, recalls, f1s):
    print("Average Scores:")
    print("Avg. accuracy=", np.sum(accuracies)/len(accuracies))
    print("Avg. precision=", np.sum(precisions, axis=0)/len(precisions))
    print("Avg. recall=", np.sum(recalls, axis=0)/len(recalls))
    print("Avg. f1=", np.sum(f1s, axis=0)/len(f1s))

def kfCrossVal(X1, yForSk, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for index, (train_ind, test_ind) in enumerate(kf.split(X1, yForSk)):
        xtrain, xtest = X1[train_ind], X1[test_ind]
        ytrain, ytest = list(yForSk[train_ind]), list(yForSk[test_ind])
        
        clf=LinearSVC(C=1.0, multi_class='crammer_singer', max_iter=1000)
        clf.fit(xtrain, ytrain)
        ypred = clf.predict(xtest)
        acc = np.mean(ytest == ypred)
        stats = precision_recall_fscore_support(ytest, ypred, labels=['-1', '0', '1'])
        
        p = np.array([stats[0][2], stats[0][1], stats[0][0]])
        r = np.array([stats[1][2], stats[1][1], stats[1][0]])
        f = np.array([stats[2][2], stats[2][1], stats[2][0]])
        
        accuracies.append(acc)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    
    printScores(accuracies, precisions, recalls, f1s)

def readTestSet(file_path):
    with open(file_path) as file:
        file.readline()
        for line in file:
            data = line.split(",")
            data[2] = preprocess(data[2])
            data[1] = preprocess(data[1], data[2])
            testSents.append(preprocess(' '.join(data[1]), ' '.join(data[2]), True))
            testDS.append(data)


def buildTestData():
    xtest = []
    sp = SenticPhrase('')

    for i, data in enumerate(testDS):
        data1 = []

        lexScore = 0
        window = 5
        atPos = int(testSents[i][1].rsplit('-', maxsplit=1)[1])
        for lind in range(atPos - window, atPos):
            if lind >= 0:
                if data[1][lind] in posLex:
                    lexScore += 1
                elif data[1][lind] in negLex or data[1][lind] in negWords:
                    lexScore -= 1
        
        for lind in range(atPos + 1, atPos + window + 1):
            if lind >= len(data[1]):
                break
            else:
                if data[1][lind] in posLex:
                    lexScore += 1
                elif data[1][lind] in negLex or data[1][lind] in negWords:
                    lexScore -= 1         

        senScore = sp.get_polarity(testSents[i][0])        
        protoVec = np.append(testXtidf[i].A[0], [lexScore, senScore])
        
        xtest.append(protoVec)

    xtest = np.array(xtest)
    return xtest

if __name__ == '__main__':

    train_path = input("Enter the path to training dataset: ")
    pred_path = input("Enter the prediction output file name: ")
    if len(train_path.strip()) == 0 or len(pred_path.strip()) == 0:
        exit()
    
    print("Reading Training DataSet and preprocessing...")
    readTrainDataSet(train_path)
    print("Reading Lexicons...")
    readLexicons()

    Xtidf = word2vec.fit_transform(list(np.array(sentences)[:, 0]))

    print("Building input training features...")
    X1, yForSk = buildTrainingData()

    flag = input("Do you want to run 10-fold-cross-validation? (y/n) : ")
    if flag == 'y':
        kfCrossVal(X1, yForSk, 10)

    print("Training the model...")
    clf=LinearSVC(C=1.0, multi_class='crammer_singer', max_iter=10000)
    clf.fit(X1, yForSk)

    test_path = input("Enter the path to test dataset: ")
    if len(test_path.strip()) == 0:
        exit()

    print("Reading Test DataSet and preprocessing...")
    readTestSet(test_path)
    testXtidf = word2vec.transform(list(np.array(testSents)[:, 0]))

    print("Building input test features...")
    xtest = buildTestData()

    print("Predicting test set...")
    ypred = clf.predict(xtest)

    print("Writing to output file...")
    with open(pred_path, 'a') as ff:
        for i in range(len(testDS)):
            ff.write(testDS[i][0]+";;"+ypred[i]+'\n')
