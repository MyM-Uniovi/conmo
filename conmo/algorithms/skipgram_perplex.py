import string
import pandas as pd
import numpy as np

from scipy.stats import chi2
from typing import Any, Iterable, Union
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.models import Model

from conmo.conf import Index, Label
from conmo.algorithms.algorithm import AnomalyDetectionThresholdBasedAlgorithm

class SkipGramPerplexity(AnomalyDetectionThresholdBasedAlgorithm):

    def __init__(self, embed_size: int, epochs: int, threshold_mode: str = 'chi2', threshold_value: Union[int, float, None] = 0.95):
        super().__init__(threshold_mode, threshold_value)
        self.embed_size = embed_size
        self.epochs = epochs

    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame: 
        # Load train corpus and generate skip-grams
        data_train_corpus = self.dataframe_to_corpus(data_train)
        skip_grams, word2id, wids = self.generate_skipgrams(data_train_corpus, False)

        # Calculate vocabulary size
        vocab_size = len(word2id) + 1

        # Build neural network architecture
        self.model = self.build_nn(vocab_size)

        # Fit model with TRAIN data (skip-grams)
        train_loss = self.fit(skip_grams)

        # Calculate cutoff (anomaly_threshold)
        anomaly_threshold = self.find_anomaly_threshold(
            train_loss, data_train.shape[1])

        # Load test corpus
        data_test_corpus = self.dataframe_to_corpus(data_test)

        # Diagnose anomalies
        pred = self.predict(data_test_corpus, wids, word2id)

        # Create arranged index
        lvl_0 = [1]
        lvl_1 = [x for x in range(1, 28479)]
        new_index = pd.MultiIndex.from_product([lvl_0, lvl_1], names=['sequence', 'time'])
        
        # Detect anomalies
        pred = pd.DataFrame(
            pred, index=new_index, columns=['distance'])
        pred.loc[:, Label.ANOMALY] = pred.loc[:, 'distance'] > anomaly_threshold

        pred.loc[(1, 28479), 'anomaly'] = False

        # Generate output dataframe
        if self.labels_per_sequence(labels_test):
            # Only labels per SEQUENCE
            output = pred.groupby(level=Index.SEQUENCE)[
                Label.ANOMALY].any()
        else:
            # Labels per TIME
            output = pred.loc[:, Label.ANOMALY]
        output = pd.DataFrame(output, index=pred.index, columns=[
                              Label.ANOMALY])
        return output
         
    def dataframe_to_corpus(self, data: pd.DataFrame) -> str:
        # Convert Dataframe to a large string containing all words
        corpus = []
        smd_text = "" # Empty string
        # Fill corpus
        for i in data.loc[:,'discretized_data']:
            smd_text += i
            smd_text += " "

        corpus.append(smd_text)

        return corpus

    def generate_skipgrams(self, corpus: str, debug: bool):
        # Create and fit tokenizer with corpus
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(corpus)

        # Create dictionaries with relationship between Ids and words
        word2id = tokenizer.word_index
        id2word = {v:k for k, v in word2id.items()}

        vocab_size = len(word2id) + 1 

        wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in corpus]

        print('Vocabulary size:', vocab_size)
        print('Most frequent words:', list(word2id.items())[-5:])

        # Generate skip-grams
        skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=5) for wid in wids]

        # Show some skip-grams
        if debug:
            pairs, labels = skip_grams[0][0], skip_grams[0][1]
            for i in range(10):
                print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
                    id2word[pairs[i][0]], pairs[i][0], 
                    id2word[pairs[i][1]], pairs[i][1], 
                    labels[i]))

        return skip_grams, word2id, wids

    def build_nn(self, vocab_size: int):
        word_model = Sequential()
        word_model.add(Embedding(vocab_size, self.embed_size,
                                embeddings_initializer="glorot_uniform",
                                input_length=1))
        word_model.add(Reshape((self.embed_size, )))

        context_model = Sequential()
        context_model.add(Embedding(vocab_size, self.embed_size,
                        embeddings_initializer="glorot_uniform",
                        input_length=1))
        context_model.add(Reshape((self.embed_size,)))

        merged_output = dot([word_model.output, context_model.output],axes=1)
        model_combined = Sequential()
        model_combined.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))
        final_model = Model([word_model.input, context_model.input], model_combined(merged_output))
        final_model.compile(loss="mean_squared_error", optimizer="rmsprop")

        # Print summary
        print(final_model.summary())

        return final_model

    def fit(self, skip_grams):
        for epoch in range(0, self.epochs):
            loss = 0
            for i, elem in enumerate(skip_grams):
                pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
                pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
                labels = np.array(elem[1], dtype='int32')
                X = [pair_first_elem, pair_second_elem]
                Y = labels
                if i % 10000 == 0:
                    print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
                loss += self.model.train_on_batch(X,Y)  

            print('Epoch:', epoch, 'Loss:', loss)
        
        return loss

    def predict(self, data_test_corpus, wids, word2id):
        # Some test words do not seem to have appeared in train
        for w in text.text_to_word_sequence(data_test_corpus[0]):
            if w not in word2id:
                word2id[w] = len(word2id)+1    
            wids.append(word2id[w])
        
        # The context is the previous point
        X_diag = [
            np.array(wids[0][1:],dtype='int32'),
            np.array(wids[0][:-1], dtype='int32')]

        res = self.model.predict(X_diag)

        # TODO: Estimate perplexity
        # Now this is working with probabilities

        return res
    
    def find_anomaly_threshold(self, values: np.ndarray, n_features: int) -> float:
        if self.threshold_mode == 'chi2':
            return chi2.ppf(self.threshold_value, df=n_features)
        else:
            super().find_anomaly_threshold(values)
    
