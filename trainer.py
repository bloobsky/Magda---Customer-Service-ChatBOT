import nltk

import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words= []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', ':']
data_file = open('navigation.json').read()
intents = json.loads(data_file)


for intent in intents['data']:
    for pattern in intent['patterns']:

        #Tokenize word (Splitting sentences into individual words)
        to_word = nltk.word_tokenize(pattern)
        words.extend(to_word)

        #Adding words into documents 
        documents.append((to_word, intent['tag']))
       
        #Adding classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmatize (group together) and lower each word and remove duplicates or remove ignore_words
words = [lemmatizer.lemmatize(to_word.lower()) for to_word in words if to_word not in ignore_words]

#Sorting the words
words = sorted(list(set(words)))

#Sorting Classes
classes = sorted(list(set(classes)))

#Documents is a combination between patterns and intents
#print (len(documents), "documents")
# classes = intents
#print (len(classes), "classes", classes)
#print (len(words), "unique lemmatized words", words)


# Pickle module is responsible for transfering string into a binary code that could be understand by computer
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))

# Array for training data
training = []
# Array for our output
output = [0] * len(classes)

# It's a collection of words to represent a sentence with word count and mostly disregarding the order in which they appear
for doc in documents:
    bag = []
    #Tokenized words for the pattern
    pattern_words = doc[0]
    #Lemmatization
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Creating BOW (Bag of Words) array with 1 when word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output = 0 tag = 1 (for each pattern)
    output_row = list(output)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle data and putting it inside numpy array which is used for scientific data, has to have dtype=object otherwise will return ValueError
random.shuffle(training)
training = np.array(training, dtype=object)

#Create train lists:  X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
#print("Training data is )

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# that are equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fit and save the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)