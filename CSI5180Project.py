import spacy
import numpy as np
import pandas as pd 

# read the train data
data_train = pd.read_csv('/Volumes/KUN/Python/CSI5180Project/atis_train.csv', header=None)

# read the test data
data_test = pd.read_csv('/Volumes/KUN/Python/CSI5180Project/atis_test.csv', header=None)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# select data of the top 8 categories of intents
# train_after: 4834*2
# test_after: 800*2
data_train.columns = ['text','intent']
data_train = data_train.query('intent=="abbreviation" | intent=="aircraft" | intent=="airfare" | intent=="airline" | intent=="flight" | intent=="flight_time" | intent=="ground_service" | intent=="quantity"')
data_train = data_train.reset_index(drop=True)

data_test.columns = ['text','intent']
data_test = data_test.query('intent=="abbreviation" | intent=="aircraft" | intent=="airfare" | intent=="airline" | intent=="flight" | intent=="flight_time" | intent=="ground_service" | intent=="quantity"')
data_test = data_test.reset_index(drop=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load the spacy model en_core_web_lg as nlp
# https://spacy.io/models/en
# 685k keys, 685k unique vectors (300 dimensions)
nlp = spacy.load('en_core_web_lg')

# show the dimensionality of nlp.vocab
model_embedding_dim = nlp.vocab.vectors_length
#300

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Encoding Sentences Using spaCy NLP Model
# the number of sentences (row), model_embedding_dim (column) 
# original datasets
# train 4978*300, test 893*300 
# selected datasets
# train 4834*300, test 800*300

def encode_sentences(st):
    # Calculate number of sentences
    n_st = len(st)
    #print('Number of sentences:', n_st)

    # initial a new array of given shape and type, filled with zeros
    X = np.zeros((n_st, model_embedding_dim))

    # Iterate over all the sentences
    for idx, sen in enumerate(st):
        # Pass each sentence to the nlp object to create a document
        doc = nlp(sen)
        # Save the document's .vector attribute to the corresponding row in X
        # transfer a sentence to a array with 300 dimensionality 
        X[idx, :] = doc.vector
    return X

data_train.columns = [0,1]
data_test.columns = [0,1]

sentences_train = data_train[0]
sentences_test = data_test[0]

train_X = encode_sentences(sentences_train)
test_X = encode_sentences(sentences_test)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Label Encoding - intent
def label_encoding(lb):
    # Calculate the length of labels
    #n_labels = len(lb)
    #print('Number of labels:', n_labels)

    # import labelencoder
    from sklearn.preprocessing import LabelEncoder

    # instantiate labelencoder object
    le = LabelEncoder()
    le.fit(lb)
    print("list:", list(le.classes_))
    target = list(le.classes_)
    y = le.transform(lb)
    #print('Length of y:',y.shape)
    return y, target

labels_train = data_train[1]
labels_test = data_test[1]

from collections import Counter
Counter(labels_train)
Counter(labels_test)

train_y, target_name = label_encoding(labels_train)
test_y, target_name_test = label_encoding(labels_test)
y_true = test_y.copy()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# output csv files about train_X, train_y, test_X, test_y
train_X = pd.DataFrame(train_X)
train_y = pd.DataFrame(train_y)
test_X = pd.DataFrame(test_X)
test_y = pd.DataFrame(test_y)


train_X.to_csv("/Volumes/KUN/Python/CSI5180Project/preprocessing/train_X.csv",index=False, header=False)
train_y.to_csv("/Volumes/KUN/Python/CSI5180Project/preprocessing/train_y.csv",index=False, header=False)
test_X.to_csv("/Volumes/KUN/Python/CSI5180Project/preprocessing/test_X.csv",index=False, header=False)
test_y.to_csv("/Volumes/KUN/Python/CSI5180Project/preprocessing/test_y.csv",index=False, header=False)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Intent classification with SVM 
# training
from sklearn.svm import SVC
from sklearn.metrics import classification_report

train_y = np.array(train_y)
clf = SVC(C=1)
model = clf.fit(train_X, train_y.ravel())

y_pred = model.predict(test_X)
print("target_name:", target_name)
print(classification_report(y_true, y_pred,target_names=target_name))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Intent classification with neural_network.MLPClassifier 
# training
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

train_y = np.array(train_y)
clf = MLPClassifier(random_state=1, max_iter=300)
model = clf.fit(train_X, train_y.ravel())

y_pred = model.predict(test_X)
print("target_name:",target_name)
print(classification_report(y_true, y_pred,target_names=target_name))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# input a sentence and return its intent
while True:
    print("\n")
    print("Input quit or QUIT to exit! \n")
    input_sentence = input("please input a sentence:")
    if input_sentence == 'quit' or input_sentence == 'QUIT':
        break;
    U = np.zeros((1,300))
    #print("U1:",U)
    U = pd.DataFrame(nlp(input_sentence).vector)
    #print("U2:",U)
    U = U.T
    #print("U3:",U)
    y_pred = model.predict(U)
    print("\n")
    print("the intent of the sentence is:", target_name[y_pred[0]])
    input_sentence = ""
