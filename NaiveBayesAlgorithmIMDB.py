"""
    The following is a solution to correctly classify unseen movie reviews as positive or negative.

    Algorithm: Multinomial Naive Bayes learning algorithm.

    Dataset: movie reviews from IMDB (Internet Movie Database).

    Author: Darren Smith
"""

import time
import os
import math
import nltk
import HTMLParser
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# stop_words = set(stopwords.words("english"))
# english_vocab = set(w.lower() for w in nltk.corpus.words.words())
# ps = PorterStemmer()
posWords = set(line.strip() for line in open("DictionaryPosNeg/ListOfPosWords.txt", 'r'))
negWords = set(line.strip() for line in open("DictionaryPosNeg/ListOfNegWords.txt", 'r'))


def cleandata(file_contents):

    # # split words by space
    # words = file_contents.split()

    # # HTML parser
    # file_contents = file_contents.decode('utf8')
    # html_parser = HTMLParser.HTMLParser()
    # file_contents = html_parser.unescape(file_contents)

    # # remove punctuation
    # regex = re.compile('[%s]' % re.escape(string.punctuation))
    # file_contents = regex.sub(' ', file_contents)

    # remove everything not a letter
    file_contents = re.sub('[^a-zA-Z]+', ' ', file_contents)

    # lowercase everything
    file_contents = file_contents.lower()

    # split string into words
    words = nltk.word_tokenize(file_contents)

    # # stop words
    # filtered_sentence = []
    # for w in words:
    #     if w not in stop_words:
    #         filtered_sentence.append(w)

    # # english vocab
    # english_words = []
    # for w in filtered_sentence:
    #     if w in english_vocab:
    #         english_words.append(w)

    # # stem words
    # stem_words = []
    # for w in english_words:
    #     w = ps.stem(w)
    #     if w not in stem_words:
    #         stem_words.append(w)

    # # remove words 2 or less long
    # short_words = []
    # for w in stem_words:
    #     if len(w) > 2:
    #         short_words.append(w)

    # positive and negative words
    neg_pos_words = []
    for w in words:
        if w in negWords:
            neg_pos_words.append(w)
        if w in posWords:
            neg_pos_words.append(w)

    return neg_pos_words


# Start the timer
start_time = time.time()

# Global variables used
uniqueWords = {}  # The number of instances of the word. Format = word: [neg, pos]
calculations = {}  # The probability of each word. Format = word: [neg, pos]
probabilityNegFile = 0.0  # The probability file is negative
probabilityPosFile = 0.0  # The probability file is positive
totalNegCount = 0  # The total number of words in the negative files
totalPosCount = 0  # The total number of words in the positive files

# Get lists of the files
fileListNeg = os.listdir("LargeIMDB/neg")
fileListPos = os.listdir("LargeIMDB/pos")

# Calculate probability file is pos/neg
probabilityNegFile = (float(len(fileListNeg))) / (len(fileListNeg) + len(fileListPos))
probabilityPosFile = (float(len(fileListPos))) / (len(fileListNeg) + len(fileListPos))

# Get unique words in neg
for file in fileListNeg:
    file_object = open("LargeIMDB/neg/" + file, "r")
    file_contents = file_object.read()
    file_object.close()

    # Clean data and count words
    words = cleandata(file_contents)
    totalNegCount += len(words)

    for word in words:
        if word not in uniqueWords:
            uniqueWords[word] = [1, 0]
        else:
            uniqueWords[word][0] += 1

# Get unique words in pos
for file in fileListPos:
    file_object = open("LargeIMDB/pos/" + file, "r")
    file_contents = file_object.read()
    file_object.close()

    # Clean data and count words
    words = cleandata(file_contents)
    totalPosCount += len(words)

    for word in words:
        if word not in uniqueWords:
            uniqueWords[word] = [0, 1]
        else:
            uniqueWords[word][1] += 1

# Calculate the probabilities
for word in uniqueWords:
    negCalculation = (float(uniqueWords[word][0]) + 1) / (totalNegCount + len(uniqueWords))
    posCalculation = (float(uniqueWords[word][1]) + 1) / (totalPosCount + len(uniqueWords))
    calculations[word] = [negCalculation, posCalculation]

print "Number of words in dictionary:" + str(len(calculations))

# -----------------------------------
# --------------TESTING--------------
# -----------------------------------

# Global variables used

negGuessNeg = 0
posGuessNeg = 0
negGuessPos = 0
posGuessPos = 0

# Get lists of the test files
testFileListNeg = os.listdir("smallTest/neg")
testFileListPos = os.listdir("smallTest/pos")

# Get unique words in neg
for file in testFileListNeg:
    file_object = open("smallTest/neg/" + file, "r")
    file_contents = file_object.read()
    file_object.close()

    # Clean data
    words = cleandata(file_contents)

    # Calculate probability
    probabilityOfTestFileNeg = math.log(probabilityNegFile)
    probabilityOfTestFilePos = math.log(probabilityPosFile)

    # Predict if file is neg/pos
    for word in words:
        if word in calculations:
            probabilityOfTestFileNeg += math.log(calculations[word][0])
            probabilityOfTestFilePos += math.log(calculations[word][1])

    if probabilityOfTestFileNeg > probabilityOfTestFilePos:
        negGuessNeg += 1
    else:
        posGuessNeg += 1

# Get unique words in pos
for file in testFileListPos:
    file_object = open("smallTest/pos/" + file, "r")
    file_contents = file_object.read()
    file_object.close()

    # Clean data
    words = cleandata(file_contents)

    # Calculate probability
    probabilityOfTestFileNeg = math.log(probabilityNegFile)
    probabilityOfTestFilePos = math.log(probabilityPosFile)

    # Predict if file is neg/pos
    for word in words:
        if word in calculations:
                probabilityOfTestFileNeg += math.log(calculations[word][0])
                probabilityOfTestFilePos += math.log(calculations[word][1])

    if probabilityOfTestFileNeg < probabilityOfTestFilePos:
        posGuessPos += 1
    else:
        negGuessPos += 1

print "Total files: " + str(len(testFileListNeg) + len(testFileListPos))
print "Correct Neg File Guesses: " + str(negGuessNeg)
print "Wrong Neg File Guesses: " + str(posGuessNeg)
print "Correct Pos File Guesses: " + str(posGuessPos)
print "Wrong Pos File Guesses: " + str(negGuessPos)
print "Total Percentage of Correct Guesses: " + str(
    (float(negGuessNeg + posGuessPos) / (len(testFileListNeg) + len(testFileListPos))) * 100) + "%"

print ("--- %s seconds ---" % (time.time() - start_time))
