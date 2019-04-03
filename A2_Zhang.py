import re #python’s regular expression package
import numpy as np


def tokenize(sent):

    p = re.compile(r'((?:[A-Z]\.)+|(?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', re.UNICODE)
    tokens = p.findall(sent)
    return tokens


def getFeaturesForTarget(tokens, targetI, wordToIndex, nounlist):
    #input: tokens: a list of tokens, 
    #       targetI: index for the target token
    #       wordToIndex: dict mapping ‘word’ to an index in the feature list. 
    #output: list (or np.array) of k feature values for the given target

    #<FILL IN>

    word = tokens[targetI]
    isCapital = word[0].isupper()

    if(isCapital == True):
        cal = [1]        
    else: 
        cal = [0]
    capital = np.array(cal)
    firstTarget = np.zeros(257)
    if(ord(word[0]) < 256):
        firstTarget[ord(word[0])] = 1
    else: 
        firstTarget[256] = 1   # one-hot value for the first letter

    #append the previous word's one hot value
    prev = np.zeros(len(wordToIndex)+1)
    isPrevNoun = np.zeros(1)
    isFirst = np.zeros(1)
    if ( targetI == 0 ):
        isFirst[0] = 1
    else:
        prevword = tokens[targetI-1]
        try:
            prevIndex = wordToIndex[prevword]
            prev[prevIndex] = 1
        except Exception:
            prev[len(wordToIndex)] = 1
        try:
            if(prevword in nounlist):
                isPrevNoun[0] = 1
        except Exception:
            pass

    #append current to vector
    isCurNoun = np.zeros(1)
    try:
        current = wordToIndex[word]
        cur = np.zeros(len(wordToIndex)+1)
        cur[current] = 1

    except Exception:
        cur = np.zeros(len(wordToIndex)+1)
        cur[len(wordToIndex)] = 1
    try:
        if(current in nounlist):
            isCurNoun[0] = 1
    except Exception:
        pass

    #append next to vector
    nextWord = np.zeros(len(wordToIndex)+1)
    isNextNoun = np.zeros(1)
    if (targetI == len(tokens) - 1):
        pass
    else:
        nextword = tokens[targetI+1]
        try:
            nextIndex = wordToIndex[nextword]
            nextWord[nextIndex] = 1
        except Exception:
            nextWord[len(wordToIndex)] = 1
        try:
            if(nextword in nounlist):
                isNextNoun[0] = 1
        except Exception:
            pass

    features = np.concatenate((capital, firstTarget, np.array([len(word)]), isFirst, prev, isPrevNoun, cur, isCurNoun, nextWord, isNextNoun))
    return features


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def trainTagger(features, tags):
    # inputs: features: feature vectors (i.e. X)
    #        tags: tags that correspond to each feature vector (i.e. y)
    # output: model -- a trained (i.e. fit) sklearn.lienear_model.LogisticRegression model
    # print(features[:3], tags[:3])

    # train different models and pick the best according to a development set:
    Cs = [.001, .01, .1, 1, 10, 100, 1000, 10000]
    penalties = ['l1', 'l2']
    X_train, X_dev, y_train, y_dev = train_test_split(features, tags,
                                                      test_size=0.10,
                                                      random_state=42)
    bestAcc = 0.0
    bestModel = None
    for pen in penalties:  # l1 or l2
        for c in Cs:  # c values:
            model = LogisticRegression(random_state=42, penalty=pen, multi_class='auto', \
                                       solver='liblinear', C=c)
            model.fit(X_train, y_train)
            modelAcc = metrics.accuracy_score(y_dev, model.predict(X_dev))
            if modelAcc > bestAcc:
                bestModel = model
                bestAcc = modelAcc

    print("Chosen Best Model: \n", bestModel, "\nACC: %.3f" % bestAcc)

    return bestModel

from sklearn import metrics
from collections import Counter


def testAndPrintAcurracies(tagger, features, true_tags):
    #inputs: tagger: an sklearn LogisticRegression object to perform tagging
    #        features: feature vectors (i.e. X)
    #        true_tags: tags that correspond to each feature vector (i.e. y)     
    #no returns but prints the accuracy and precision, recall, f1 for each tag
    # (accuracy provided, look into metrics package for other metrics)
    
    pred_tags = tagger.predict(features)
    print("\nModel Accuracy: %.3f" % metrics.accuracy_score(true_tags, pred_tags))
    #most Frequent Tag: 
    mfTags = [Counter(true_tags).most_common(1)[0][0]]*len(true_tags) 
    print("MostFreqTag Accuracy: %.3f" % metrics.accuracy_score(true_tags, mfTags))

    return




def getConllTags(filename):
    #input: filename for a conll style parts of speech tagged file
    #output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f: 
            wordtag=wordtag.strip()
            if wordtag:#still reading current sentence
                # (word, tag) = wordtag.split("\t")
                inputtuple = wordtag.split("\t")
                if len(inputtuple) == 2:
                    (word, tag) = inputtuple
                else:
                    (word, tag) = (" ", inputtuple[0])
                wordTagsPerSent[sentNum].append((word, tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent  





from sys import argv

corpus1 = 'daily547.conll'
corpus2 = 'oct27.conll'
nouns = 'cap.1000'
sampleSentences = \
    ['The horse raced past the barn fell.',
     'For 4 years, we attended S.B.U. in the CS program.',
     'Did you hear Sam tell me to "chill out" yesterday? #rude',
     'He told Barak Obama to read Newsday to learn about Stony Brook University']

if __name__ == "__main__":
        
    if len(argv) > 1:#replace with argument for filename if available
        try:
            get_ipython()
        except: #not in python notebook; use argv
            corpus = argv[1]
    
    ###########################################
    #1) Test The Tokenizer
    for sent in sampleSentences:
        print(sent, "\n", tokenize(sent), "\n")
    
    ###########################################
    #2) Run Feature Extraction:
    #2a) load training data: 
    wordToIndex = set()
    nounlist = list()
    taggedSents = getConllTags(corpus1)+getConllTags(corpus2)
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            wordToIndex |= set(words) #union of the words into the set
    print("[Read ", len(taggedSents), " Sentences]")
    with open(nouns) as fp:
        line = fp.readline()
        while line:
            nounlist += list(line.strip())
            line = fp.readline()
    fp.close()

    #make dictionaries for converting words to index and tags to ids:
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)} 

    #2b) Call feature extraction on each target
    X = []
    y = []
    print("[Extracting Features]")
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            for i in range(len(words)):
                y.append(1 if tags[i] == '^' else 0) #append y with class label
                X.append(getFeaturesForTarget(words, i, wordToIndex, nounlist))
    X, y = np.array(X), np.array(y)
    print("[Done X is ", X.shape, " y is ", y.shape, "]")

    
    ####################################################
    #3 Train the model. 
    print("[Training the model]")
    tagger = trainTagger(X, y)
    print("[done]")
    
    ###################################################
    #4 Test the tagger.
    testAndPrintAcurracies(tagger, X, y)
    
    ###################################################
    #5 Apply to example sentences:
    print("\n[Applying to sample sentences]")
    for sent in sampleSentences:
        tokens = tokenize(sent)
        sentX = []
        for i in range(len(tokens)):
            sentX.append(getFeaturesForTarget(tokens, i, wordToIndex, nounlist))
        pred_tags = tagger.predict(sentX)
        sentWithTags = zip(tokens, pred_tags)
        print(sent, "\n  predicted tags: ", list(sentWithTags))
     
