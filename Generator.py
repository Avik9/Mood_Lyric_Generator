import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scipy.stats
import sys
import pandas as pd

##################################################################
# 1. Get song lyric corpus and tokenize it


def tokenize(sent):
    """
    Tokenizes each word and returns the token.

    Attributes
    ----------
    sent : str
        A string representing the lyrics of a song.

    Returns
    -------
    [] : list
        A list of tokenized words in the text.

    Steps
    -----
    Step 1.3: Tokenize the song titles (The tokenizer).
    Step 1.4: Tokenize the song lyrics (The tokenizer).

    """
    return re.findall(r"(?:\w(?:\w*[-&'%$]\w+)+)|(?:\[\w+\])|(?:(?:<+[/A-Za-z0-9]*>+)|(?:[_#@]*[a-z0-9]+\b)|(?:[A-Z\.!?#$%&\,-`]+[A-Za-z0-9]*))", sent)


def readCSV(fileName, counterLimit=None):
    """
    Reads in all the files and stores them for future use.

    Attributes
    ----------
    fileName : str
        The name of the csv file to open.
    counterLimit : int
        The number of songs to use to make the dictionary.

    Returns
    -------
    songs : dict
        The dictionary containing all the songs.

    Steps
    -----
    Step 1.2: Read the csv into memory.
    Step 1.3: Tokenize the song titles (calls the tokenizer).
    Step 1.4: Tokenize the song lyrics (calls the tokenizer).

    """
    counterLimit = 100000 if counterLimit is None else counterLimit

    songs = {}

    dataFile = pd.read_csv(fileName, sep=',')

    for _, line in dataFile.iterrows():

        if counterLimit <= 0:
            break

        lyrics = "<s> " + line[3].replace("\n", " <newline> ") + " </s>"
        # Step 1.3: Tokenize the song titles.
        title_tokenized = [w for w in tokenize(line[1])]
        # Step 1.4: Tokenize the song lyrics.
        lyrics_tokenized = [w for w in tokenize(lyrics)]
        artist_song = ((line[0] + "-" + line[1]).lower()).replace(" ", "_")
        songs[artist_song] = [title_tokenized, lyrics_tokenized]

        counterLimit -= 1

    return songs

##################################################################
# 2: Code an add1-trigram language model method


def createVocab(songs):
    """
    Creates a dictionary of all the words that appear more than 2 times.

    Attributes
    ----------
    songs : dict
        The dictionary containing all the songs.

    Returns
    -------
    finalVocab : dict
        The dictionary containing all the tokenized words and their count.

    Steps
    -----
    Step 2.1: Create a vocabulary of words from lyrics.

    """

    vocab = {}
    finalVocab = {}

    for value in songs.values():
        for word in value[1]:
            word = word.lower()
            vocab[word] = 1 if word not in vocab else vocab[word] + 1

    finalVocab["<oov>"] = 0

    for key in vocab:
        if vocab[key] > 2:
            finalVocab[key] = vocab[key]
        else:
            finalVocab["<oov>"] += 1

    return finalVocab


def createMatrixCounts(finalVocab, songs):
    """
    Creates a dictionary of all the words that appear more than 2 times.

    Attributes
    ----------
    finalVocab : dict
        The dictionary containing all the tokenized words.
    songs : dict
        The dictionary containing all the songs.

    Returns
    -------
    [] : list
        The dictionary containing the bigramCounts and trigramCounts.

    Steps
    -----
    Step 2.2: Create a bigram matrix (rows as previous word; columns as current word).
    Step 2.3: Create a trigram matrix (rows as previous bigram; columns as current word).

    """

    bigramCounts = {}
    trigramCounts = {}

    for value in songs.values():

        lyrics = value[1]
        lyricsLen = len(lyrics)
        for pos in range(lyricsLen+1):

            # Step 2.2: Create a bigram matrix (rows as previous word; columns as current word).
            if pos > 0 and pos != lyricsLen:
                prev_word = lyrics[pos - 1].lower()
                curr_word = lyrics[pos].lower()

                prev_word = "<oov>" if prev_word not in finalVocab else prev_word
                curr_word = "<oov>" if curr_word not in finalVocab else curr_word

                if prev_word in bigramCounts:
                    if curr_word in bigramCounts[prev_word]:
                        bigramCounts[prev_word][curr_word] += 1
                    else:
                        bigramCounts[prev_word][curr_word] = 1
                else:
                    bigramCounts[prev_word] = {curr_word: 1}

            # Step 2.3: Create a trigram matrix (rows as previous bigram; columns as current word).
            if pos > 1 and pos != lyricsLen:
                prev_prev_word = lyrics[pos - 2].lower()
                prev_word = lyrics[pos - 1].lower()
                curr_word = lyrics[pos].lower()

                prev_prev_word = "<oov>" if prev_prev_word not in finalVocab else prev_prev_word
                prev_word = "<oov>" if prev_word not in finalVocab else prev_word
                curr_word = "<oov>" if curr_word not in finalVocab else curr_word

                if (prev_prev_word, prev_word) in trigramCounts:
                    if curr_word in trigramCounts[(prev_prev_word, prev_word)]:
                        trigramCounts[(prev_prev_word, prev_word)
                                      ][curr_word] += 1
                    else:
                        trigramCounts[(prev_prev_word, prev_word)
                                      ][curr_word] = 1
                else:
                    trigramCounts[(prev_prev_word, prev_word)] = {curr_word: 1}
            if pos == lyricsLen:
                bigramCounts[lyrics[pos - 1]] = {}
                trigramCounts[(lyrics[pos - 2], lyrics[pos - 1])] = {}

    return [bigramCounts, trigramCounts]


def generateProbabilities(words, vocab, bigramCounts, trigramCounts):
    """
    Calculate the probability of all possible current words wi
    given either a single previous word or two previous words.

    Attributes
    ----------
    words : list
        The list containing wi-2, wi-1, and wi.
    vocab : dict
        The dictionary containing all the tokenized words.
    bigramCounts : dict
        Contains the count of how many times a word wi occurs after wi-1.
    trigramCounts : dict
        Contains the count of how many times a word wi occurs after wi-1 and wi-2.

    Returns
    -------
    probDict : dict
        A dictionary of words and their probability of wi occuring after wi-1.

    Steps
    -----
    Step 2.4: Create a method to calculate the probability of all possible 
                current words wi  given either a single previous word 
                (wi-1 -- a bigram model) or two previous words (wi-1 and 
                wi-2 -- a trigram model).
    """

    wordDict = {}
    probDict = {}

    word1 = words[2]
    word2 = words[1]
    wordPred = words[0]

    if word1 not in bigramCounts:
        word1 = "<oov>"

    wordDict = bigramCounts[word1]
    tempDict = {}
    for word in wordDict:
        if word is not "<oov>":
            tempDict[word] = wordDict[word]

    wordDict = tempDict

    if len(wordDict) == 0:
        # generate unigram probs
        num_words = getUnigramTotal(vocab)
        for token in vocab:
            if token is not "<oov>":
                probDict[token] = vocab[token] / num_words
        return probDict

    for token in wordDict:
        if token is not "<oov>":
            if word2 is not None:
                probDict[token] = calculateProbabilty(
                    [token, word1, word2], vocab, bigramCounts, trigramCounts)
            else:
                probDict[token] = calculateProbabilty(
                    [token, word1], vocab, bigramCounts, trigramCounts)

    if wordPred == None:
        # predict words and their probabilities without wi
        return probDict
    else:
        # predict words and their probabilities including wi
        if wordPred not in probDict and wordPred is not "<oov>":
            probDict[wordPred] = vocab[wordPred] / num_words
        return probDict

    return probDict


def calculateProbabilty(words, vocab, bigramCounts, trigramCounts):
    """
    Calculates the probability of wi given either wi or wi-1 or wi-2.

    Attributes
    ----------
    words : list
        The list containing wi-2, wi-1, and wi.
    vocab : dict
        The dictionary containing all the tokenized words.
    bigramCounts : dict
        Contains the count of how many times a word Wi occurs after Wi-1.
    trigramCounts : dict
        Contains the count of how many times a word Wi occurs after Wi-1 and Wi-2.

    Returns
    -------
    probability : float
        The probability of wi occuring after wi-1 or wi-1 and wi-2.

    Steps
    -----
    Step 2.4: Create a method to calculate the probability of all possible 
                current words wi  given either a single previous word 
                (wi-1 -- a bigram model) or two previous words (wi-1 and 
                wi-2 -- a trigram model).
    """

    bigramProb = 0
    # Bigram formula: P_add1(Xi | Xi-1) = count(Xi-1 Xi) + 1 / count(Xi-1) + vocab(Xi-1)

    if len(words) == 2:
        word1 = words[(len(words)) - 1].lower()     # Last word = "love"
        wordProb = words[0].lower()                 # Word to predict = "you"

    else:
        wordProb = words[(len(words)) - 1].lower()  # Last word = "love"
        word1 = words[1].lower()                    # Second to last word = "I"

    bigramWordCount = 1

    word1Present = 1 if word1 in bigramCounts else 0

    if word1Present and wordProb in bigramCounts[word1]:
        bigramWordCount += bigramCounts[word1][wordProb]

    elif not word1Present and wordProb in bigramCounts["<oov>"]:
        word1 = "<oov>"
        bigramWordCount += bigramCounts[word1][wordProb]

    elif word1Present and wordProb not in bigramCounts[word1]:
        wordProb = "<oov>"
        bigramWordCount = 1

    bigramProb = bigramWordCount/(len(bigramCounts[word1]) + vocab[word1])
    unigramProb = vocab[wordProb] / getUnigramTotal(vocab)

    if(len(words) == 2):
        return ((bigramProb + unigramProb) / 2)

    trigramProb = 0
    # Trigram Formula: P_add1(w3 | w1, w2) = trigramCounts[(w1, w2)][w3] + 1 / bigramCounts[(w1, w2)] + VOCAB_SIZE

    word2 = words[1].lower()                    # Second to last word = "I"
    word1 = words[(len(words)) - 1].lower()     # Last word = "love"
    wordProb = words[0].lower()                 # Word to predict = "you"

    trigramWordCount = 1
    wordsPresent = 1 if (word2, word1) in trigramCounts else 0

    if wordsPresent and wordProb in trigramCounts[(word2, word1)]:
        trigramWordCount += trigramCounts[(word2, word1)][wordProb]

    elif wordsPresent and wordProb not in trigramCounts[(word2, word1)]:
        if "<oov>" in trigramCounts[(word2, word1)]:
            trigramWordCount += trigramCounts[(word2, word1)]["<oov>"]

    elif not wordsPresent:
        word2 = "<oov>" if word2 not in vocab else word2
        word1 = "<oov>" if word1 not in vocab else word1

        if (word2, word1) in trigramCounts:
            if wordProb in trigramCounts[(word2, word1)]:
                trigramWordCount += trigramCounts[(word2, word1)][wordProb]

    trigramProb = trigramWordCount / (bigramWordCount - 1 + len(vocab))
    unigramProb = vocab[wordProb] / getUnigramTotal(vocab)

    return (trigramProb + bigramProb + unigramProb) / 3


def getUnigramTotal(vocab):
    """
    Returns the total occurance of each key in a given dictionary.

    Attributes
    ----------
    vocab : dict
        The vocab dictionary.

    Returns
    -------
    counter : int
        The total occurance of each key in a given dictionary.

    Steps
    -----
    Step 2.4: Create a method to calculate the probability of all possible 
              current words wi  given either a single previous word 
              (wi-1 -- a bigram model) or two previous words (wi-1 and 
               wi-2 -- a trigram model).
    Step 3.5: Create a method to generate lyrics given an adjective 
              (i.e. the language that was trained on lyrics for titles with the 
              given adjective).

    """
    counter = 0
    for word in vocab:
        counter += vocab[word]

    return counter

##################################################################
# 3: Create adjective-specific language models

    ######################### FROM HW I ######################


vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y']
punctuation = list(r"!\"\#$%&'()*+, -./:;<=>?@[\]^_`{|}~")
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def getFeaturesForTokens(tokens, wordToIndex):
    """
    Given a sentence as a list of tokens, returns a list of feature vectors per token.

    Attributes
    ----------
    tokens : tokens
        The list containing tokenized words.
    wordToIndex : dict
        Dict mapping 'word' to an index in the feature list.

    Returns
    -------
    featuresPerTarget : list
        A list of lists (or n * m np.array where n = number of tokens 
        and m = len(wordToIndex) * 3 + 2) of k feature values for the given target.

    Steps
    -----
    Step 3.1: Train your model based on your assignment 1 code.

    """

    num_words = len(tokens)
    featuresPerTarget = list()  # holds arrays of feature per word

    for targetI in range(num_words):

        currentList = list()
        currentList = [0] * ((len(wordToIndex) + 1) * 3)

        if(targetI == 0 and targetI + 1 == num_words):  # Only word
            pass

        elif(targetI == 0 and targetI + 1 < num_words):  # First word
            # Next word is in the right most []
            if wordToIndex.get(tokens[targetI + 1].lower()) != None:
                currentList[wordToIndex[tokens[targetI + 1].lower()] +
                            (len(wordToIndex) + 1) * 2] = 1
            else:
                currentList[((len(wordToIndex) + 1) * 3) - 1] = 1  # OOV Word

        elif(targetI == num_words - 1 and targetI - 1 >= 0):  # Last word

            # Previous word is in the left most []
            if wordToIndex.get(tokens[targetI - 1].lower()) != None:
                currentList[wordToIndex[tokens[targetI - 1].lower()]] = 1
            else:
                currentList[len(wordToIndex) + 1] = 1  # OOV Word
        else:
            # Previous word is in the left most []
            if wordToIndex.get(tokens[targetI - 1].lower()) != None:
                currentList[wordToIndex[tokens[targetI - 1].lower()]] = 1
            else:
                currentList[len(wordToIndex) + 1] = 1  # OOV Word

            # Next word is in the right most []
            if wordToIndex.get(tokens[targetI + 1].lower()) != None:
                currentList[wordToIndex[tokens[targetI + 1].lower()] +
                            len(wordToIndex)*2] = 1
            else:
                currentList[((len(wordToIndex) + 1) * 3) - 1] = 1  # OOV Word

        # Current word is in the middle []
        if wordToIndex.get(tokens[targetI].lower()) != None:
            currentList[(wordToIndex[tokens[targetI].lower()] +
                         len(wordToIndex))] = 1
        else:
            currentList[((len(wordToIndex) + 1) * 2) - 1] = 1  # OOV Word

        currentList.append(numberOfConstants(tokens[targetI]))
        currentList.append(numberOfConstants(tokens[targetI], False))
        featuresPerTarget.append(currentList)

    return featuresPerTarget  # a (num_words x k) matrix


def numberOfConstants(word, constants=True):
    """
    Given a word, returns the number of consonants and vowels.

    Attributes
    ----------
    word : str
        A single token.
    constants : Boolean
        Boolean telling us to find the number of consonants in True or number of vowels if False.

    Returns
    -------
    count : int
        The number of vowels / consonants in the word

    Steps
    -----
    Step 3.1: Train your model based on your assignment 1 code.

    """
    count = 0

    for letter in word:
        if constants:
            count += 1 if letter not in punctuation and letter not in vowels and letter not in numbers else 0

        else:
            count += 1 if letter in vowels else 0

    return count


def trainAdjectiveClassifier(features, adjs):
    """
    Builds a logistic regression model that classifies words in a sentence as being either adjectives or not, given the feature vectors.

    Attributes
    ----------
    features : list
        Feature vectors (i.e. X).
    adjs : list
        Whether adjective or not: [0, 1] (i.e. y).

    Returns
    -------
    accurate_model : sklearn.linear_model.LogisticRegression
        A trained sklearn.linear_model.LogisticRegression object

    Steps
    -----
    Step 3.1: Train your model based on your assignment 1 code.

    """

    listC = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    maxAccuracy = 0
    bestC = 1000

    X_train, X_test, Y_train, Y_test = train_test_split(
        features, adjs, test_size=0.20, random_state=42)

    for C in listC:

        model = LogisticRegression(
            penalty="l1", solver='liblinear', C=C, random_state=42).fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        # compute accuracy:
        leny = len(Y_test)
        acc = np.sum([1 if (y_pred[i] == Y_test[i])
                      else 0 for i in range(leny)]) / leny

        if acc >= maxAccuracy:
            maxAccuracy = acc
            bestC = C
    return bestC


def getConllTags(filename):
    """
    Given a word, returns the number of consonants and vowels.

    Attributes
    ----------
    filename : str
        Name for a conll style parts of speech tagged file.

    Returns
    -------
    wordTagsPerSent : sklearn.linear_model.LogisticRegression
        A list of list of tuples representing [[[word1, tag1], [word2, tag2]]]

    Steps
    -----
    Step 3.1: Train your model based on your assignment 1 code.

    """

    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent


def getAdjectives(songs):
    """
    Given a list of song titles, returns the predicted adjectives that occur at least 10 times.

    Attributes
    ----------
    songs : dict
        The dictionary contains all the songs.

    Returns
    -------
    [] : list
        A list of adjectives that were predicted at least 10 times and the count of 
        how many times each adjective was predicted.

    Steps
    -----
    Step 3.1: Train your model based on your assignment 1 code
    Step 3.2: Extract features for adjective classifier
    Step 3.3: Find adjectives in each title.

    """

    # Step 3.1: Train your model based on your assignment 1 code
    taggedSents = getConllTags('daily547.conll')

    wordToIndex = set()  # maps words to an index
    for sent in taggedSents:
        if sent:
            # splits [(w, t), (w, t)] into [w, w], [t, t]
            words, tags = zip(*sent)
            # union of the words into the set
            wordToIndex |= set([w.lower() for w in words])

    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    sentXs = []
    sentYs = []
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex))
            sentYs.append([1 if t == 'A' else 0 for t in tags])

    from sklearn.model_selection import train_test_split
    X = [j for i in sentXs for j in i]
    Y = [j for i in sentYs for j in i]
    try:
        X_train, _, Y_train, _ = train_test_split(
            np.array(X), np.array(Y), test_size=0.10, random_state=42)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)

    bestC = trainAdjectiveClassifier(X_train, Y_train)

    trained_model = LogisticRegression(
        penalty="l1", solver='liblinear', C=bestC, random_state=42).fit(X, Y)

    adjectives = {}
    adjectives_count = {}

    counter = 1

    for song in songs:
        counter += 1
        title_tokenized = songs[song][0]
        # Step 3.2: Extract features for adjective classifier
        title_tokenized = [w.lower() for w in title_tokenized]
        features = getFeaturesForTokens(title_tokenized, wordToIndex)
        prediction = trained_model.predict(features)

        # Step 3.3: Find adjectives in each title
        for pos in range(len(prediction)):
            if prediction[pos] == 1:
                if title_tokenized[pos] in adjectives:
                    adjectives_count[title_tokenized[pos]] += 1
                    adjectives[title_tokenized[pos]][song] = songs[song]
                else:
                    adjectives_count[title_tokenized[pos]] = 1
                    adjectives[title_tokenized[pos]] = {song: songs[song]}

    tempDict = {}
    for key in adjectives_count:
        if(adjectives_count[key] > 10):
            tempDict[key] = adjectives_count[key]

    adjectives_count = tempDict

    return [adjectives, adjectives_count]


def generateLanguage(adjective_model):
    """
    Given a list of vocab dict, bigram matrix, and trigram matrix for a 
    specific adjective, returns a sentence generated from that adjective.

    Attributes
    ----------
    adjective_model : list
        The list contains the vocab dict, bigram matrix, and trigram matrix
        for a specific adjective.

    Returns
    -------
    sentence : str
        Lyrics generated given an adjective.

    Steps
    -----
    Step 3.5: Create a method to generate lyrics given an adjective 
    (i.e. the language that was trained on lyrics for titles with the 
    given adjective).

    """

    vocabDict = adjective_model[0]
    bigramMatrix = adjective_model[1]
    trigramMatrix = adjective_model[2]

    max_length = 32
    word_generated = ""  # you
    word2 = ""  # love
    word1 = ""  # i
    sentence = ["<s>"]

    while max_length > 0 and word_generated is not "</s>":
        possible_probs = []
        possible_words = []

        if len(sentence) > 1:
            word2 = sentence[len(sentence) - 2]
            word1 = sentence[len(sentence) - 1]

            possible_wi = generateProbabilities(
                [None, word2, word1], vocabDict, bigramMatrix, trigramMatrix)
            possible_probs = list(possible_wi.values())
            possible_words = list(possible_wi)

        else:
            word1 = sentence[0]
            possible_wi = generateProbabilities(
                [None, None, word1], vocabDict, bigramMatrix, trigramMatrix)
            possible_probs = list(possible_wi.values())
            possible_words = list(possible_wi)

        if "<oov>" in possible_words:
            temp_dict = {}
            for index in range(len(possible_probs)):
                if possible_words[index] is not "<oov>":
                    temp_dict[possible_words[index]] = possible_probs[index]

            possible_wi = temp_dict

            if len(possible_wi) == 0:
                # generate unigram probs
                num_words = getUnigramTotal(vocabDict)
                possible_wi = {}
                for token in vocabDict:
                    if token is not "<oov>":
                        possible_wi[token] = vocabDict[token] / num_words

            possible_probs = list(possible_wi.values())
            possible_words = list(possible_wi)

        possible_probs = np.divide(possible_probs, np.sum(possible_probs))

        choice = np.random.choice(
            a=len(possible_words), replace=False, p=possible_probs)
        key_counter = 0

        for key in possible_words:
            word_generated = key if key_counter == choice else word_generated
            key_counter += 1
        sentence.append(word_generated)

        max_length -= 1

    sentence.append("</s>") if "</s>" not in sentence else ""

    return " ".join(sentence)

##################################################################
##################################################################
# Main:


if __name__ == '__main__':

    # Stage 1: Get song lyric corpus and tokenize it

    print("\nStage 1 Checkpoint:")
    print()

    # Step 1.2: Read the csv into memory.
    processedSongs = readCSV('songdata.csv')
    test = ["abba-burning_my_bridges", "beach_boys-do_you_remember?",
            "avril_lavigne-5,_4,_3,_2,_1_(countdown)", "michael_buble-l-o-v-e"]

    # Stage 1 Checkpoint
    for test_song in test:
        print("Printing", test_song, ":")
        print("Tokenized Title:", processedSongs[test_song][0])
        print("Tokenized Lyrics:", processedSongs[test_song][1:])
        print()

    # Stage 2: Code an add1-trigram language model method

    print("\n\nStage 2 Checkpoint: Based on just the first 5,000 lyrics, print the following (add-1 smoothed) probabilities")
    print()

    # Step 2.1: Create a vocabulary of words from lyrics.
    processedSongs2 = readCSV('songdata.csv', 5000)

    vocabDict = createVocab(processedSongs2)

    # Step 2.2: Create a bigram matrix (rows as previous word; columns as current word)
    # Step 2.3: Create a trigram matrix (rows as previous bigram; columns as current word)
    matrices = createMatrixCounts(vocabDict, processedSongs2)

    # Stage 2 Checkpoint

    print('p(wi = "you"     | wi-2 = "I",           wi-1 = "love")      :',
          calculateProbabilty(["you", "I", "love"],               vocabDict, matrices[0], matrices[1]))
    print('p(wi = "special" |                       wi-1 = "midnight")  :',
          calculateProbabilty(["special", "midnight"],            vocabDict, matrices[0], matrices[1]))
    print('p(wi = "special" |                       wi-1 = "very")      :',
          calculateProbabilty(["very", "special"],                vocabDict, matrices[0], matrices[1]))
    print('p(wi = "special" | wi-2 = "something",   wi-1 = "very")      :',
          calculateProbabilty(["special", "something", "very"],   vocabDict, matrices[0], matrices[1]))
    print('p(wi = "funny"   | wi-2 = "something",   wi-1 = "very")      :',
          calculateProbabilty(["funny", "something", "very"],     vocabDict, matrices[0], matrices[1]))

    # Stage 3: Create adjective-specific language models

    print("\nFinal Checkpoints:")

    test = ["good", "happy", "afraid", "red", "blue"]

    # Step 3.3: Find adjectives in each title.
    temp_list = getAdjectives(processedSongs)

    adjectives = temp_list[0]
    adjective_count = temp_list[1]
    adjective_models = {}

    # Step 3.4: Build a separate language model for the lyrics associated with each
    # adjective from the previous step.
    for adj in adjectives:
        vocabulary = createVocab(adjectives[adj])
        matrices = createMatrixCounts(vocabulary, adjectives[adj])
        adjective_models[adj] = [vocabulary, matrices[0], matrices[1]]

    Final Checkpoint
    for i in range(len(test)):
        if test[i] in adjectives:
            print()
            print("All artist-songs associated with", test[i], ":")
            print(list(adjectives[test[i]]))

            sentence = ""
            for x in range(1, 4):

                # Step 3.5: Create a method to generate lyrics given an adjective
                # (i.e. the language that was trained on lyrics for titles with the given adjective).
                sentence = generateLanguage(adjective_models[test[i]])
                print()
                print(test[i], ":", sentence)
