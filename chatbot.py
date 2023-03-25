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
                to_word = nltk.word_tokenize(pattern)
                self.words.extend(to_word)
                self.documents.append((to_word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = sorted(list(set([self.lemmatizer.lemmatize(to_word.lower()) for to_word in self.words if to_word not in self.ignore_words])))
        self.classes = sorted(list(set(self.classes)))
        
        pickle.dump(self.words, open('texts.pkl', 'wb'))
        pickle.dump(self.classes, open('labels.pkl', 'wb'))

    def create_training_data(self):
        training = []
        output = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            output_row = list(output)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        training = np.array(training, dtype=object)
        self.train_x = list(training[:,0])
        self.train_y = list(training[:,1])

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.train_y[0]), activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train_model(self, epochs, batch_size):
        self.hist = self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, filename):
        self.model.save(filename, self.hist)


if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.load_data('navigation.json')
    chatbot.create_training_data()
    chatbot.create_model()
    chatbot.train_model(500, 5)
    chatbot.save_model('model.h5')