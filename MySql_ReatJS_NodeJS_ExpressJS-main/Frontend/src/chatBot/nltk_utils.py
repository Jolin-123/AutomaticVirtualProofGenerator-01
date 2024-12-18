
import nltk
import numpy as np

# Download 'punkt' for tokenization
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag =    [ 0,     1,     0,     1,      0,     0,       0 ]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bow = bag_of_words(sentence, words)

print(bow)



# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)


# words = ["Organize", "Organizes", "Organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)