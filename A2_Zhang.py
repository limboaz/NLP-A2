import re  # python’s regular expression package
import numpy as np


def tokenize(sent):
    p = re.compile(r'((?:[A-Z]\.)+|(?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', re.UNICODE)
    tokens = p.findall(sent)
    return tokens


def getFeaturesForTarget(tokens, targetI, wordToIndex, nounlist):
    # input: tokens: a list of tokens,
    #       targetI: index for the target token
    #       wordToIndex: dict mapping ‘word’ to an index in the feature list. 
    # output: list (or np.array) of k feature values for the given target

    # <FILL IN>

    word = tokens[targetI]
    isCapital = word[0].isupper()

    if (isCapital == True):
        cal = [1]
    else:
        cal = [0]
    capital = np.array(cal)
    firstTarget = np.zeros(257)
    if (ord(word[0]) < 256):
        firstTarget[ord(word[0])] = 1
    else:
        firstTarget[256] = 1  # one-hot value for the first letter

    # append the previous word's one hot value
    prev = np.zeros(len(wordToIndex) + 1)
    isPrevNoun = np.zeros(1)
    isFirst = np.zeros(1)
    if (targetI == 0):
        isFirst[0] = 1
    else:
        prevword = tokens[targetI - 1]
        try:
            prevIndex = wordToIndex[prevword]
            prev[prevIndex] = 1
        except Exception:
            prev[len(wordToIndex)] = 1
        try:
            if (prevword in nounlist):
                isPrevNoun[0] = 1
        except Exception:
            pass

    # append current to vector
    isCurNoun = np.zeros(1)
    try:
        current = wordToIndex[word]
        cur = np.zeros(len(wordToIndex) + 1)
        cur[current] = 1

    except Exception:
        cur = np.zeros(len(wordToIndex) + 1)
        cur[len(wordToIndex)] = 1
    try:
        if (current in nounlist):
            isCurNoun[0] = 1
    except Exception:
        pass

    # append next to vector
    nextWord = np.zeros(len(wordToIndex) + 1)
    isNextNoun = np.zeros(1)
    if (targetI == len(tokens) - 1):
        pass
    else:
        nextword = tokens[targetI + 1]
        try:
            nextIndex = wordToIndex[nextword]
            nextWord[nextIndex] = 1
        except Exception:
            nextWord[len(wordToIndex)] = 1
        try:
            if (nextword in nounlist):
                isNextNoun[0] = 1
        except Exception:
            pass

    features = np.concatenate(
        (capital, firstTarget, np.array([len(word)]), isFirst, prev, isPrevNoun, cur, isCurNoun, nextWord, isNextNoun))
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


def baseModel(targetCounts, grams, vocabulary):
    ngramCounts = targetCounts
    ngramModelProbs = dict()  # stores p(Xi|Xi-1), [x--k...x-1][xi]
    for ngram, count in ngramCounts.items():
        p = count / grams[ngram[0:-1]]
        try:
            ngramModelProbs[ngram[0:-1]][ngram[-1]] = p  # indexed by [x--k...x-1][xi]
        except KeyError:
            ngramModelProbs[ngram[0:-1]] = {ngram[-1]: p}
    return ngramModelProbs


from sklearn import metrics
from collections import Counter


def testAndPrintAcurracies(tagger, features, true_tags):
    # inputs: tagger: an sklearn LogisticRegression object to perform tagging
    #        features: feature vectors (i.e. X)
    #        true_tags: tags that correspond to each feature vector (i.e. y)     
    # no returns but prints the accuracy and precision, recall, f1 for each tag
    # (accuracy provided, look into metrics package for other metrics)

    pred_tags = tagger.predict(features)
    print("\nModel Accuracy: %.3f" % metrics.accuracy_score(true_tags, pred_tags))
    # most Frequent Tag:
    mfTags = [Counter(true_tags).most_common(1)[0][0]] * len(true_tags)
    print("MostFreqTag Accuracy: %.3f" % metrics.accuracy_score(true_tags, mfTags))

    return


def getConllTags(filename):
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                # (word, tag) = wordtag.split("\t")
                inputtuple = wordtag.split("\t")
                if len(inputtuple) == 2:
                    (word, tag) = inputtuple
                else:
                    (word, tag) = (" ", inputtuple[0])
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent


import tweepy
import sys
from time import sleep


class StreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        self.tweets = []
        super().__init__(api)

    def on_status(self, status):
        tweet = status._json['retweeted_status'] if 'retweeted_status' in status._json.keys() else status._json

        if 'extended_tweet' in tweet.keys():
            text = tweet['extended_tweet']['full_text']
        else:
            text = status.text

        self.tweets.append(text)

    def on_error(self, status):
        print(status)
        return False

    def get_tweets(self):
        return self.tweets


def retrieve_tweets(named_entity, total, key, secret):
    # Get app secrets
    # secrets = open("secrets.txt")
    # hard coded the key and secret for the sake of submitting one file
    consumer_key = "szIHs77Towpo9k3PeGw7UOcql"
    consumer_secret = "ED1q3SrkX5vGLwi6cY089M5s35CNprrfwY0CDjEs5UDPL3m0dg"
    # secrets.close()

    # oauth2
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(key, secret)
    api = tweepy.API(auth)

    # Start Stream
    stream_listener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener, tweet_mode='extended')
    stream.filter(track=[named_entity], is_async=True)

    while len(stream_listener.get_tweets()) < total:
        sleep(1)

    stream.disconnect()
    return stream_listener.get_tweets()


def NamedEntityGenerativeSummary(named_entity, twitter_access_token, twitter_access_token_secret):
    # train the named entity recognizer; save it to an object, train the generic (base) trigram language model; save it
    recognizer, wordCounts, bigramCounts, trigramCounts, taggedSents, wordToIndex, nounlist = trainRecognizer()

    # pull 1000 tweets that contain named_entity
    tweets = retrieve_tweets(named_entity, 1000, twitter_access_token, twitter_access_token_secret)

    # Limit to tweets with the named entity being classified as a named entity
    taggedTweets = []
    tweetvocabulary = set()
    for tweet in tweets:
        tokens = tokenize(tweet)
        tweetvocabulary |= set(tokens)
        tweetX = []
        if len(tokens) == 0:
            continue

        for i in range(len(tokens)):
            tweetX.append(getFeaturesForTarget(tokens, i, wordToIndex, nounlist))
        pred_tags = recognizer.predict(tweetX)
        tweetWithTags = zip(tokens, pred_tags)
        taggedTweets += tweetWithTags

        words = tokens
        for i in range(len(tokens)):
            try:
                wordCounts[(words[i],)] += 1
            except KeyError:
                wordCounts[(words[i],)] = 1

            # count the bigram
            if (i > 0):
                bigram = (words[i - 1], words[i])
                try:
                    bigramCounts[bigram] += 1
                except KeyError:
                    bigramCounts[bigram] = 1

            # count the trigrams
            if (i > 1):
                trigram = (words[i - 2], words[i - 1], words[i])
                try:
                    trigramCounts[trigram] += 1
                except KeyError:
                    trigramCounts[trigram] = 1

    vocab = tweetvocabulary | wordToIndex.keys()
    trigram_model = baseModel(trigramCounts, bigramCounts, vocab)
    bigram_model = baseModel(bigramCounts, wordCounts, vocab)

    # generate five different phrases that follow the named_entity.
    named_entity_tokens = tokenize(named_entity)
    phraseCount = 0
    while phraseCount < 5:
        last_bigram = named_entity_tokens[-2:]  # last two words of named entity
        new_word = named_entity_tokens[-1]
        phrase = named_entity
        i = 0
        while i < 5 and new_word != "END":
            # get probability distribution for next word based on last two words
            try:
                choices = trigram_model[tuple(last_bigram)].items()
            except KeyError:
                choices = bigram_model[tuple(last_bigram[-1:])].items()
            # pick a next word from the probability distribution
            words = []
            probabilities = []
            for word, prob in choices:
                words.append(word)
                probabilities.append(prob)

            try:
                new_word = np.random.choice(a=words, p=probabilities)
            except:
                new_word = np.random.choice(a=words)
            # update last_bigram = (last_bigram[1], new_word)
            last_bigram = (last_bigram[-1], new_word)
            phrase += " " + new_word
            i += 1
        print(phrase)
        phraseCount += 1




from sys import argv


def trainRecognizer():
    corpus1 = 'daily547.conll'
    corpus2 = 'oct27.conll'
    nouns = 'cap.1000'
    sampleSentences = \
        ['The horse raced past the barn fell.',
         'For 4 years, we attended S.B.U. in the CS program.',
         'Did you hear Sam tell me to "chill out" yesterday? #rude',
         'He told Barak Obama to read Newsday to learn about Stony Brook University']

    nounlist = list()
    counts = dict()
    wordCounts = dict()
    bigramCounts = dict()
    trigramCounts = dict()

    taggedSents = getConllTags(corpus1) + getConllTags(corpus2)
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            for word in words:
                try:
                    counts[word] += 1
                except KeyError:
                    counts[word] = 1
    wordToIndex = [word for word in counts.keys() if counts[word] > 1]
    print("[Read ", len(taggedSents), " Sentences]")
    with open(nouns) as fp:
        line = fp.readline()
        while line:
            nounlist += line.strip()
            line = fp.readline()
    fp.close()

    # iterate through each sentence, and extract word and bigram counts
    for sent in taggedSents:
        words = [word.lower() for word, tag in sent]  # grabbing words, droppin gtags
        # print("\nNext Sent:", words)
        for i in range(len(words)):
            try:
                wordCounts[(words[i],)] += 1
            except KeyError:
                wordCounts[(words[i],)] = 1

            # count the bigram
            if (i > 0):
                bigram = (words[i - 1], words[i])
                try:
                    bigramCounts[bigram] += 1
                except KeyError:
                    bigramCounts[bigram] = 1

            # count the trigrams
            if (i > 1):
                trigram = (words[i - 2], words[i - 1], words[i])
                try:
                    trigramCounts[trigram] += 1
                except KeyError:
                    trigramCounts[trigram] = 1

    # make dictionaries for converting words to index and tags to ids:
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    # 2b) Call feature extraction on each target
    X = []
    y = []
    print("[Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            for i in range(len(words)):
                y.append(1 if tags[i] == '^' else 0)  # append y with class label
                X.append(getFeaturesForTarget(words, i, wordToIndex, nounlist))
    X, y = np.array(X), np.array(y)
    print("[Done X is ", X.shape, " y is ", y.shape, "]")

    ####################################################
    # 3 Train the model.
    print("[Training the model]")
    tagger = trainTagger(X, y)
    testAndPrintAcurracies(tagger, X, y)
    return tagger, wordCounts, bigramCounts, trigramCounts, taggedSents, wordToIndex, nounlist

    ###################################################
    # 4 Test the tagger.


    ###################################################
    # 5 Apply to example sentences:
    # print("\n[Applying to sample sentences]")
    # for sent in sampleSentences:
    #     tokens = tokenize(sent)
    #     sentX = []
    #     for i in range(len(tokens)):
    #         sentX.append(getFeaturesForTarget(tokens, i, wordToIndex, nounlist))
    #     pred_tags = tagger.predict(sentX)
    #     sentWithTags = zip(tokens, pred_tags)
    #     print(sent, "\n  predicted tags: ", list(sentWithTags))

# these are my own tokens for the sake of submitting one file
token = "4200770712-lOSJRmsCZZSBX2OWFDRIc5RzSKzut4PyIsdsYDR"
secret = "4eRO6pUhaz5dZpQSRTtE1fNiIRGOvQHSaIU3k4qN5h5wD"

NamedEntityGenerativeSummary('Trump', token, secret)
