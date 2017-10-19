import random;
import codecs;
import nltk;
import string;
import gensim;
from nltk.stem.porter import PorterStemmer

random.seed(123);


f = codecs.open('bokSMAL.txt', "r", "utf-8");


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
    return tempArray;


def stem(array):
    stemmer = PorterStemmer();
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j] = stemmer.stem(array[i][j]);
    return array;

paragraphs = makeParagraphs(f);
removeGutenberg = removeWord("Gutenberg", paragraphs);
print(len(paragraphs))
print(len(removeGutenberg))

tokenized = tokenize(removeGutenberg);
noPunctuation = removePunctuationAndLower(tokenized);


