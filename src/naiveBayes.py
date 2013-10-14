'''
File : naiveBayes.py
'''
'''
Author : Jagan Mohan Rao Vujjini
'''
'''
ID : 800804731
'''

from random import shuffle
import matplotlib.pyplot as plt 
import operator
import math
import re
import time

def process(line):
    '''process each line of the file and return a list of processed strings removing special characters'''
    
    result = []
    line = line.strip()
    line = re.sub('[^\w]', ' ', line)
    line = line.lower()
    result = line.split()

    return result    

def readFile(file):
    
    '''Reads the file into a list'''
    
    dataSet = []
    
    for line in file:
        dataSet.append(line)
    
    shuffle(dataSet)
    
    return dataSet

def breakDataSet(dataSet):
    
    '''Breaks the DataSet into 5 parts for 5-Fold Cross Validation'''
    
    splitDataSet = []
    splitValue = len(dataSet)/5.0
    count = 0
    
    while float(count) < len(dataSet):
        splitDataSet.append(dataSet[int(count):int(float(count)+splitValue)])
        count += splitValue
    
    return splitDataSet

def prob(word,vocablaryList,totalLabel,totalSize,label):
    
    '''calculates the probability of each word based on the training set. If the word is not present,
    it assumes that the word exists once(m-estimate)'''
    
    spamWord = 1
    hamWord = 1
    
    if word in vocablaryList.keys():
        spamWord += vocablaryList[word]['spam']
        hamWord += vocablaryList[word]['ham']
        
    prob2 = totalLabel/float(totalSize)
    prob3 = (spamWord+hamWord)/float(totalSize)
    
    if label == 'spam':                
        prob1 = spamWord/float(totalLabel)
    else:
        prob1 = hamWord/float(totalLabel)
    
    return (prob1*prob2)/prob3
    
def trainNaiveBayes(splitDataSet, validationSetIndex):
    
    '''trains our classifier and breaks the file into dictionary of words with an estimate of how many times 
    the word appears in Spam, Ham and the Total Number of occurences'''
    
    msgWordList = []
    vocablaryList = dict()
    spamDict = dict()
    hamDict = dict()
    testSet = []
    trainingSetSize = 0
    trainingSpamSize = 0
    trainingHamSize = 0
    totalSpamWords = 0
    totalHamWords = 0
    
    for i in xrange(5):
        
        if i != validationSetIndex:
            
            dataSet = splitDataSet[i]
            
            for message in dataSet:
                trainingSetSize += 1
                splitLine = message.split("\t")
                msgWordList = process(splitLine[1])
                if splitLine[0] == 'ham':
                    trainingHamSize += 1
                    for word in msgWordList:
                        if word in vocablaryList:
                            vocablaryList[word]['ham'] +=1
                            vocablaryList[word]['total'] +=1
                            totalHamWords +=1
                        else:
                            vocablaryList[word] = {'ham': 1, 'spam': 0, 'total' : 1}
                            totalHamWords += 1
                else:
                    trainingSpamSize += 1
                    for word in msgWordList:
                        if word in spamDict:
                            vocablaryList[word]['spam'] +=1
                            vocablaryList[word]['total'] +=1
                            totalSpamWords += 1
                        else:
                            vocablaryList[word] = {'ham': 0, 'spam': 1, 'total' : 1}
                            totalSpamWords += 1
        
        else:
            testSet = splitDataSet[i]
    
    return trainingSetSize,trainingSpamSize,trainingHamSize,vocablaryList,totalSpamWords,totalHamWords,testSet

def testNaiveBayes(number,trainingSetSize,trainingSpamSize,trainingHamSize,vocablaryList,spamWordCount,hamWordCount,testSet):
    
    '''Tests each Message in the validation set with the training set and assigns it a argmax of probability of 
    spam and ham. We also calculate tp,fp,tn,tn. To counter the zero values we use summation of log's of 
    probabilities instead of product of probabilities'''
    
    result = dict()
    result['spam'] = 0
    result['ham'] = 0
    result['TruePositives'] = 0
    result['TrueNegatives'] = 0
    result['FalsePositives'] = 0
    result['FalseNegatives'] = 0
    result['trainingSize'] = trainingSetSize
    
    sortedSpamList = sorted(vocablaryList.iteritems(), key=lambda (x, y): y['total'],reverse=True)
    #sorted(vocablaryList, key=vocablaryList['Total'].get, reverse=True)
    
    for i in xrange(number):
        del vocablaryList[sortedSpamList[i][0]]
    
    for message in testSet:
        splitLine = message.split("\t")
        msgWordList = process(splitLine[1])
        prob_word_spam = 0.0
        prob_word_ham = 0.0
        spam_probability = math.log(trainingSpamSize/float(trainingSetSize))
        ham_probability = math.log(trainingHamSize/float(trainingSetSize))
        for word in msgWordList:
            prob_word_spam = prob(word,vocablaryList,spamWordCount,trainingSetSize,'spam')
            spam_probability += math.log(prob_word_spam)  
            prob_word_ham = prob(word,vocablaryList,hamWordCount,trainingSetSize,'ham')
            ham_probability += math.log(prob_word_ham)
        if spam_probability > ham_probability:
            result['spam'] += 1
            
            if splitLine[0] == 'ham':
                result['FalseNegatives'] += 1
            else:
                result['TrueNegatives'] += 1
            
        else:
            result['ham'] += 1
            
            if splitLine[0] == 'spam':
                result['FalsePositives'] += 1
            else:
                result['TruePositives'] += 1
            
    return result

def main():
    
    '''The Program runs from here and displays the required parameters'''
    
    file = open('SMSSpamCollection', 'r')
    
    dataSet = readFile(file)
    splitDataSet = breakDataSet(dataSet)
    
    print "***********************************************************************************************************"
    print "starting 5 Fold Validation"
    print "***********************************************************************************************************"
    
    for i in range(5):
        
        print "\n stage %d of 5 Fold Validation" %(i+1)
        trainingStart = time.time()    
        trainingSetSize,trainingSpamSize,trainingHamSize,vocablaryList,totalSpamWords,totalHamWords,testSet = trainNaiveBayes(splitDataSet,i)
        trainingEnd = time.time() - trainingStart
        #print vocablaryList
    
        for num in [10,25,50,100,500]:
            print "\t removing %d most frequent words" %(num)
            testingStart = time.time()
            result = testNaiveBayes(num,trainingSetSize,trainingSpamSize,trainingHamSize,vocablaryList,totalSpamWords,totalHamWords,testSet)
            testingEnd = time.time() - testingStart
            
            print "Training Time : " + str(trainingEnd) + " secs"
            print "Testing Time : " + str(testingEnd) + " secs"
            
            correctResults = (result['TruePositives']+result['TrueNegatives'])
            incorrectResults = (result['FalsePositives']+result['FalseNegatives'])
            error = ((100*(result['FalsePositives']+result['FalseNegatives']))/float(result['trainingSize']))
            accuracy = ((100*(result['TruePositives']+result['TrueNegatives']))/float(result['TruePositives']+result['TrueNegatives']+result['FalsePositives']+result['FalseNegatives']))
            tpRate = ((100*result['TruePositives'])/float(result['trainingSize']))
            fpRate = ((100*result['FalsePositives'])/float(result['trainingSize'])) 
            precision = ((100*result['TruePositives'])/float(result['FalsePositives']+result['TruePositives']))
            recall =  ((100*result['TruePositives'])/float(result['FalseNegatives']+result['TruePositives']))
            sensitivity = ((100*result['TruePositives'])/float(result['FalseNegatives']+result['TruePositives']))
            specificity = ((100*result['TrueNegatives'])/float(result['FalsePositives']+result['TrueNegatives']))
                
            print "***********************************************************************************************************"
            print "values removing %d most frequent words" %(num)
            print "***********************************************************************************************************"
            print "Correct Results: %d" %(correctResults)
            print "InCorrectResults: %d" %(incorrectResults)
            print "Error: %.2f%%" %(error)
            print "Accuracy: %.2f%% " %(accuracy)
            print "True Positives Rate: %.2f%%" %(tpRate)
            print "False Positives Rate: %.2f%%" %(fpRate)
            print "Precision: %.2f%%" %(precision)
            print "Recall: %.2f%%" %(recall)
            print "Sensitivity: %.2f%%" %(sensitivity)
            print "Specificity: %.2f%%" %(specificity)
            print "***********************************************************************************************************"
            
            plt.plot(num,accuracy)
            
        print "validation stage %d is done" %(i+1)
        
        plt.show()  
    

main()