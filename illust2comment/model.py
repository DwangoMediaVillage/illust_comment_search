# -*- coding: utf-8 -*-
import chainer
import numpy
import niconico_chainer_models

class ImageModel(niconico_chainer_models.VggA):
    def predict(self, x_data, volatile=chainer.flag.OFF):
        x = chainer.Variable(x_data, volatile=volatile)
        return self.functions.forward(x, train=False)[0].data

    def feature(self, x_data, volatile=chainer.flag.OFF):
        x = chainer.Variable(x_data, volatile=volatile)
        return self.functions.forward(x, train=False)[1]["h15"]



class FeatureWordModel(chainer.Chain):
    def __init__(self, vocab_size, midsize, output_feature_size):
        super(FeatureWordModel, self).__init__(
            word_embed=chainer.functions.EmbedID(vocab_size, midsize),
            lstm0=chainer.links.connection.lstm.LSTM(midsize, midsize),
            lstm1=chainer.links.connection.lstm.LSTM(midsize, midsize),
            word_out_layer=chainer.functions.Linear(midsize, vocab_size),
            out_layer=chainer.functions.Linear(midsize, output_feature_size)
        )

    def predict_word(self, x):
        feature_predicted, word_predicted = self._forward(x)
        return word_predicted

    def loss_predict_word(self, x, t):
        feature_predicted, word_predicted = self._forward(x)
        loss = chainer.functions.softmax_cross_entropy(word_predicted, t)
        return loss, feature_predicted, word_predicted

    def feature(self, x):
        return self._forward(x)[0]

    def _forward(self, x):
        h = self.word_embed(x)
        if hasattr(self, "lstm0"):
            h = self.lstm0(h)
        if hasattr(self, "lstm1"):
            h = self.lstm1(h)
        if hasattr(self, "lstm2"):
            h = self.lstm2(h)
        feature = self.out_layer(h)
        word = self.word_out_layer(h)
        return feature, word

    def reset_state(self):
        if hasattr(self, "lstm0"):
            self.lstm0.reset_state()
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm1.reset_state()

class FeatureWordModel1Layer(FeatureWordModel):
    def __init__(self, vocab_size, midsize, output_feature_size):
        super(FeatureWordModel, self).__init__(
            word_embed=chainer.functions.EmbedID(vocab_size, midsize),
            lstm0=chainer.links.connection.lstm.LSTM(midsize, midsize),
            word_out_layer=chainer.functions.Linear(midsize, vocab_size),
            out_layer=chainer.functions.Linear(midsize, output_feature_size)
        )

class WordEmbedder(object):
    def __init__(self, vocabulary):
        self.embedder = {}
        self.decoder = {}
        for i, character in enumerate(vocabulary):
            if character == "\n":
                character = "[newline]"
            self.embedder[character] = i
            self.decoder[i] = character
        self.decoder[i+1] = "[start]"
        self.embedder["[start]"] = i+1
        self.decoder[i+2] = "[unknown]"
        self.embedder["[unknown]"] = i+2
        self.vecsize = len(vocabulary) + 2

    def embed_vector(self, character):
        if character == "\n":
            character = "[newline]"
        chara_index = self.embed_id(character)
        value = numpy.zeros(self.vecsize)
        value[chara_index] = 1
        return value

    def embed_id(self, character):
        if character in self.embedder:
            chara_index = self.embedder[character]
        else:
            chara_index = self.vecsize-1
        return chara_index

    def save_vocabulary(self, filename):
        with open(filename, "w+") as f:
            for i in range(self.vecsize-2):
                f.write(self.decoder[i].encode("utf-8")+"\n")
