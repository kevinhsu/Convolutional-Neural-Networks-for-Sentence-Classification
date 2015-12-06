from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from imdb import load_data, prepare_data

'''
    This example demonstrates the use of Convolution1D for text classification.
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py
    Only take sentences less than maxlen=100 words into consideration.
    Get to 0.950 test accuracy after 3 epochs. 100s/epoch on GPU.
'''

imdb_path = "imdb.pkl"
path = os.path.join(os.getcwd() + '/data/' + imdb_path)

class MyCNNSentiment:
    '''Create the imdb CNN sentiment analysis class

    This class contains four main functions to build and evaluate a convolutional neural network
    '''
    # set parameters:
    def __init__(self,path=path,max_features=100000,maxlen=100,batch_size=32,embedding_dims=100,
            nb_filter=250,filter_length=3,hidden_dims=250,nb_epoch=3,dropout_ratio=0.25,
            subsample_length=1,dense=1,border_mode="valid"):
    # '''Initialize parameters

    # :type path: String
    # :param path: The path to the dataset (here IMDB)
    # :type max_features: int
    # :param max_features: The number of maximum feature maps
    # :type maxlen: int
    # :param maxlen: The maximum length of sentences from dataset
    # :type batch_size: int
    # :param batch_size: The number of batch size
    # :type embeddings_dims: int
    # :param embeddings_dims: The dimension of word2vec vector
    # :type dropout_ratio: float32
    # :param dropout_ratio: The subsample ratio to avoid overfitting
    # :type border_mode: String
    # :param border_mode: The type pf border detection
    
    # '''
    ##################
    # INITIALIZATION #
    ##################
        self.path = path
        self.max_features = max_features
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.embedding_dims = embedding_dims
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.hidden_dims = hidden_dims
        self.nb_epoch = nb_epoch
        self.dropout_ratio = dropout_ratio
        self.subsample_length = subsample_length
        self.dense = dense
        self.border_mode = border_mode

    def generate_data(self):
        '''Load the dataset

        Generate train, valid and test dataset

        '''
        print("Loading data...")
        train, valid, _ = load_data(path=self.path)
        self.X_train, self.X_mask_train, self.Y_train = prepare_data(train[0], train[1], maxlen=self.maxlen)
        self.X_valid, self.X_mask_valid, self.Y_valid = prepare_data(valid[0], valid[1], maxlen=self.maxlen)
        del train, valid
        print(len(self.X_train), 'train sequences')
        print(len(self.X_valid), 'valid sequences')
        print("Pad sequences (samples x time)")
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.maxlen)
        self.X_valid = sequence.pad_sequences(self.X_valid, maxlen=self.maxlen)
        print('X_train shape:', self.X_train.shape)
        print('X_valid shape:', self.X_valid.shape)

    def build_model(self,activation1="tanh",activation2="tanh",activation3="sigmoid"):
        ''' build model

        Input layer: tanh
        Hidden layer: tanh
        Output layer: sigmoid 

        '''
        print('Build model...')
        self.activation1 = activation1  # input layer
        self.activation2 = activation2  # hidden layer
        self.activation3 = activation3  # output layer
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen))
        self.model.add(Dropout(self.dropout_ratio))

        # we add a Convolution1D, which will learn nb_filter rather than 2-gram
        # word group filters of size filter_length:
        self.model.add(Convolution1D(nb_filter=self.nb_filter,
                                filter_length=self.filter_length,
                                border_mode=self.border_mode,
                                activation=self.activation1,
                                subsample_length=self.subsample_length))

        # we use standard max pooling (halving the output of the previous layer):
        # need Theano 0.8.0 to run MaxPoolingXD
        try:
            model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
        except:
            print("Need Theano 0.8.0 to run MaxPoolingXD")

        # We flatten the output of the conv layer,
        # so that we can add a vanilla dense layer:
        self.model.add(Flatten())

        # We add a vanilla hidden layer:
        self.model.add(Dense(self.hidden_dims))
        self.model.add(Dropout(self.dropout_ratio))
        self.model.add(Activation(self.activation2))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(self.dense))
        self.model.add(Activation(self.activation3))

    def evaluate_loss(self,loss='binary_crossentropy',optimizer='rmsprop',class_mode='binary'):
        self.model.compile(loss=loss,
                    optimizer=optimizer,
                    class_mode=class_mode)
        self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size,
                    nb_epoch=self.nb_epoch, show_accuracy=True,
                    validation_data=(self.X_valid, self.Y_valid))

def main():
    np.random.seed(1337)  # for reproducibility
    cnn = MyCNNSentiment()
    cnn.generate_data()
    cnn.build_model()
    cnn.evaluate_loss()

if __name__ == '__main__':
    main()
