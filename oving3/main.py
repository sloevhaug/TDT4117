import random;
import codecs;
import nltk;
import string;
import gensim;
from nltk.stem.porter import PorterStemmer
from pprint import pprint

random.seed(123);


f = codecs.open('bokSMAL.txt', "r", "utf-8");
stopFile = open("stopWords.txt", "r").read().split(",");


def makeParagraphs(file):
    paragraphs = [];
    paragraph = '';
    
    for line in f.readlines():
        if not line.strip():
            if(paragraph != ""):
                paragraphs.append(paragraph);
                paragraph = "";
        else:
            paragraph += line

    return paragraphs;

def removeWord(word, array):
    tempArray = [];
    for item in array:
        if word not in item:
            tempArray.append(item);

    return tempArray;

def tokenize(array):
    tempArray = [];
    for item in array:
        tempArray.append(nltk.word_tokenize(item))
        #paragraphs.append(item.split());
    return tempArray;


def removePunctuationAndLower(array):
    remove = string.punctuation + "\n\r\t";
    tempArray = [];
    tempParagraph = [];
    if isinstance(array, list):
        for i in range(len(array)):
            for j in range(len(array[i])):
                word = array[i][j];
                newWord = ''
                for char in word:
                    if char not in remove:
                        newWord += char;
                if newWord != '':
                    tempParagraph.append(newWord.lower());

            tempArray.append(tempParagraph);
            tempParagraph = [];
    else :
        for i in range(len(array)):
            word = array[i]
            newWord = ''
            for char in word:
                if char not in remove:
                    newWord += char;
            if newWord != '':
                tempArray.append(newWord.lower());
    return tempArray;


def stem(array):
    stemmer = PorterStemmer();
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j] = stemmer.stem(array[i][j]);
    return array;

def getStopWordIDs(stopwords, dictionary):
    stopWordIDs = []
    
    for word in stopwords:
        try:
            stopWordIDs.append(dictionary.token2id[word])
        except:
            pass
    return stopWordIDs;

def convertToBagOfWords(array, dictionary):
    vector = [];
    for p in array:
        vector.append(dictionary.doc2bow(p))
    return vector;

# Dividing the file into paragraphs
paragraphs = makeParagraphs(f);

# Removing paragraphs with the word "Gutenberg"
removeGutenberg = removeWord("Gutenberg", paragraphs);

# Tokenizing the paragraphs
tokenized = tokenize(removeGutenberg);

# Removing punctuation from the array
noPunctuation = removePunctuationAndLower(tokenized);

# Create dictionary based on our stemmed list
dictionary = gensim.corpora.Dictionary(noPunctuation);

# Get the IDs for the stop words in our dictionary
stopWordIDArray = getStopWordIDs(stopFile, dictionary);

# Filter out the stopwords from our dictionary
dictionary.filter_tokens(stopWordIDArray);

# Converting our list to a bag of words.
bagOfWords = convertToBagOfWords(noPunctuation, dictionary);

# Making a TD-IDF model
tfidf_model = gensim.models.TfidfModel(bagOfWords);

# Mapping bag of words into TF-IDF weights
tfidf_corpus = tfidf_model[bagOfWords];

print(tfidf_model);
matrixSimimlarity = gensim.similarities.MatrixSimilarity(tfidf_corpus);
print(matrixSimimlarity)

'''
3.4 Repeat the above procedure for LSI model using as an input the corpus with TF-IDF weights. Set
number of topics to 100. In the end, each paragraph should be represented with a list of 100 pairs (topic-
index, LSI-topic-weight) ). Some useful code:
$ ... = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary,
num_topics=100)
$ lsi_corpus = lsi_model[...]
$ ... = gensim.similarities.MatrixSimilarity(...)
3.5 Report and try to interpret first 3 LSI topics. Some useful code:
$ lsi_model.show_topics()
'''