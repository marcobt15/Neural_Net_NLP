# Name this file assignment4.py when you submit
import math
import os

class bag_of_words_model:

  def __init__(self, directory):
    # directory is the full path to a directory containing trials through state space
    words_map = {}
    document_count = 0
    for file_name in os.listdir(directory):
      document_count+=1
      with open(os.path.join(directory, file_name), 'r') as file:
        words = file.read().strip().split(" ")
        for word in words:
          words_map[word] = words_map.get(word, 0) + 1

    self.vocabulary = list(words_map.keys())
    self.vocabulary.sort()

    self.idf = []
    for word in self.vocabulary:
      word_freq = words_map[word]
      idf_val = math.log(document_count/word_freq, 2)
      self.idf.append(idf_val)
    
    # Return nothing


  def tf_idf(self, document_filepath):
    # document_filepath is the full file path to a test document

    # Return the term frequency-inverse document frequency vector for the document
    return tf_idf_vector


  def predict(self, document_filepath, weights):
    # document_filepath is the full file path to a test document
    # weights is a list of weights for the artificial neuron

    # Return the prediction from the neural network model
    return prediction
  
bag_of_words_model("Examples\\Example0\\training_documents")