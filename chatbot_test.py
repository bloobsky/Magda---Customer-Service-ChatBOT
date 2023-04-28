import unittest
import os.path
import nltk
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

from chatbot import ChatBot

class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.chatbot = ChatBot()
        self.chatbot.load_data('navigation.json')
        self.chatbot.create_training_data()
        self.chatbot.create_model()
        
    def test_load_data(self):
        self.assertTrue(len(self.chatbot.words) > 0)
        self.assertTrue(len(self.chatbot.classes) > 0)
        self.assertTrue(len(self.chatbot.documents) > 0)
        
    def test_create_training_data(self):
        self.assertTrue(len(self.chatbot.train_x) > 0)
        self.assertTrue(len(self.chatbot.train_y) > 0)
        
    def test_create_model(self):
        self.assertIsNotNone(self.chatbot.model)
        
    def test_save_model(self):
        filename = 'chatbot_model.h5'
        self.assertTrue(os.path.isfile(filename))

if __name__ == '__main__':
    unittest.main()
