import nltk
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer


class ChatBot:
    def __init__(self):
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',', ':', 'and', 'a', 'the']
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self, filename):
        data_file = open(filename).read()
        intents = json.loads(data_file)
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                #Tokenize word (Splitting sentences into individual words)
                to_word = nltk.word_tokenize(pattern)
                self.words.extend(to_word)
                #Adding word into documents
                self.documents.append((to_word, intent['tag']))
                #Adding classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        #Lemmatize (group together), sorting and lower each word and remove duplicates or remove ignore_words          
        self.words = sorted(list(set([self.lemmatizer.lemmatize(to_word.lower()) for to_word in self.words if to_word not in self.ignore_words])))
        #Sorting Classes
        self.classes = sorted(list(set(self.classes)))
        
        # Pickle module is responsible for transfering string into a binary code that could be understand by computer
        pickle.dump(self.words, open('texts.pkl', 'wb'))
        pickle.dump(self.classes, open('labels.pkl', 'wb'))

    def create_training_data(self):
        # It’s a collection of words to represent a sentence with word count and mostly disregarding the order in which they appear
        training = []
        output = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            #Tokenize words for the pattern
            pattern_words = doc[0]
            #Lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            #Creating BOW (Bag of Words) array wwith 1 when word match found in current pattern
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            output_row = list(output)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        # Shuffle data and putting it inside numpy array which is used for scientific data, has to have dtype=object otherwise will return ValueError
        random.shuffle(training)
        training = np.array(training, dtype=object)
        #Create train lists:  X - patterns, Y - intents
        self.train_x = list(training[:,0])
        self.train_y = list(training[:,1])

    def create_model(self):
        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # that are equal to number of intents to predict output intent with softmax
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.train_y[0]), activation='softmax'))
        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train_model(self, epochs, batch_size):
        #Fit the model
        self.hist = self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, filename):
        #Save the model 
        self.model.save(filename, self.hist)


if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.load_data('navigation.json')
    chatbot.create_training_data()
    chatbot.create_model()
    chatbot.train_model(500, 5)
    chatbot.save_model('model.h5')