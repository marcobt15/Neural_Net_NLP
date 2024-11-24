# Name this file assignment4.py when you submit
import math
import os


class bag_of_words_model:
    def __init__(self, directory):
        # directory is the full path to a directory containing trials through state space
        words_map = {}
        document_count = 0
        for file_name in os.listdir(directory):
            document_count += 1
            with open(os.path.join(directory, file_name), 'r') as file:
                words = file.read().strip().split(" ")
                for word in words:
                    words_map[word] = words_map.get(word, 0) + 1

        self.vocabulary = list(words_map.keys())
        self.vocabulary.sort()

        print(f"vocab {self.vocabulary}")

        self.idf = []
        for word in self.vocabulary:
            word_freq = words_map[word]
            idf_val = math.log(document_count / word_freq, 2)
            self.idf.append(idf_val)

    def tf_idf(self, document_filepath):
        with open(document_filepath, 'r') as file:
            doc = file.read()
            words = doc.split(" ")

        count_vector = []
        for word in self.vocabulary:
            count_vector.append(self.count_based_occurance(word, doc))

        term_frequency = []
        for x in count_vector:
            term_frequency.append(x / len(words))

        tf_idf_vector = [x * y for x, y in zip(term_frequency, self.idf)]

        return tf_idf_vector

    def count_based_occurance(self, word, doc):
        return doc.split().count(word)

    def predict(self, document_filepath, weights):
        y_hat = 1 / (1 + math.exp(-sum(w * x for w, x in zip(weights, self.tf_idf(document_filepath)))))

        return y_hat


model = bag_of_words_model("Examples\\Example0\\training_documents")
assert model.predict("Examples\Example0\\test_document.txt",
                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) == 0.6193563324683053

model = bag_of_words_model("Examples\\Example1\\training_documents")
assert model.predict("Examples\Example1\\test_document.txt",
                     [1.10560059, 1.34847003, -0.62618406, 1.47942839, -0.13775678, 1.00499298, 0.90386239, -0.08970558,
                      -0.30356443, 0.91637918, 0.13094345, 0.36672302, -0.54820217]) == 0.47716625779428534

model = bag_of_words_model("Examples\\Example2\\training_documents")
assert model.predict("Examples\Example2\\test_document.txt",
                     [1.10560059, 1.34847003, -0.62618406, 1.47942839, -0.13775678, 1.00499298, 0.90386239, -0.08970558,
                      -0.30356443, 0.91637918, 0.13094345, 0.36672302, -0.54820217]) == 0.6403829012892953
