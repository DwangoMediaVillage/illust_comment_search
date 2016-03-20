# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import numpy
import chainer
import chainer.serializers
import os
import collections
import pipe
import sys

import illust2comment.utility
import illust2comment.model

parser = argparse.ArgumentParser()
parser.add_argument("comment_tsv")
parser.add_argument("init_hdf5_file")
parser.add_argument("--image_dir", required=True)
parser.add_argument("--optimizer", required=True)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--max_comment_length", type=int, default=50)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--model_output_dir", required=True)
parser.add_argument("--average_model", dest='average_model', action='store_true')
parser.add_argument("--model")
parser.add_argument("--init_random", dest='init_random', action='store_true')
parser.add_argument("--init_random_range", type=float, default=0.1)
parser.add_argument("--init_comment_model", default=None, help="初期値として使うモデル")
parser.add_argument("--vocabulary", default=None)
parser.add_argument("--hidden_unit", type=int, default=1024)
args = parser.parse_args()
print(args)

CHARACTER_START = "\n"
CHARACTER_END = ""
MINIMUM_FREQUENCY = 0 #出現回数N回未満の文字は無視する

def load_id_comments_pretty(comment_tsv):
    for content_id, comment in illust2comment.utility.load_id_comments(comment_tsv):
        ## 学習しにくいコメントは除く
        if len(comment) <= 3: # too short
            continue
        if len(comment) >= args.max_comment_length - 2: # too long
            continue
        if comment[0] == "|": #|hoge|
            continue
        if comment[0] == u"｜": #|hoge|
            continue
        if comment[0] == u"↑": #↑reply.....
            continue
        yield content_id, comment


###########
## setup
###########
try:
    os.mkdir(args.model_output_dir)
except:
    print("{} already exists".format(args.model_output_dir))

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

################
## vocabulary
################
if args.vocabulary is None:
    count_dict = collections.defaultdict(int)
    for _, comment in load_id_comments_pretty(args.comment_tsv) | pipe.take(100000):
        for character in comment:
            count_dict[character] += 1


    count_dict[CHARACTER_START] = MINIMUM_FREQUENCY + 1
    count_dict[CHARACTER_END] = MINIMUM_FREQUENCY + 1
    vocabulary = [character for character, count in count_dict.items() if count >= MINIMUM_FREQUENCY]
    print(len(vocabulary))
else:
    vocabulary = [line.rstrip().decode("utf-8") for line in open(args.vocabulary)]
character_embedder = illust2comment.model.WordEmbedder(vocabulary)
character_embedder.save_vocabulary(os.path.join(args.model_output_dir, "vocabulary.txt"))
print(len(vocabulary))
print("vocabulary size: ", character_embedder.vecsize)

################
## Models
################
if args.model == "2layer":
    comment_model = illust2comment.model.FeatureWordModel(vocab_size=character_embedder.vecsize, midsize=args.hidden_unit, output_feature_size=4096)
elif args.model == "1layer":
    comment_model = illust2comment.model.FeatureWordModel1Layer(vocab_size=character_embedder.vecsize, midsize=args.hidden_unit, output_feature_size=4096)
else:
    raise Exception("invalid model")

image_model = illust2comment.model.ImageModel(406) #nico-opendata
chainer.serializers.load_hdf5(args.init_hdf5_file, image_model.functions)
if args.gpu >= 0:
    comment_model.to_gpu()
    image_model.functions.to_gpu()
if args.optimizer == "adagrad":
    optimizer = chainer.optimizers.AdaGrad(lr=args.learning_rate)
elif args.optimizer == "adam":
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
elif args.optimizer == "sgd":
    optimizer = chainer.optimizers.SGD(lr=args.learning_rate)
elif args.optimizer == "rmsprop":
    optimizer = chainer.optimizers.RMSprop(lr=args.learning_rate)
else:
    raise Exception("invalid optimizer")
if args.init_random:
    for param in comment_model.params():
        xp = chainer.cuda.get_array_module(param.data)
        param.data[:] = xp.random.uniform(-args.init_random_range, args.init_random_range, param.data.shape)
if not args.init_comment_model is None:
    chainer.serializers.load_hdf5(args.init_comment_model, comment_model)

###########
## train
###########
optimizer.setup(comment_model)
batch = []
n = 0
for content_id, comment in load_id_comments_pretty(args.comment_tsv):
    ### load
    img_path = "{}/{}.jpg".format(args.image_dir, content_id[2:])
    if not os.path.exists(img_path):
        continue
    n += 1

    # 最大長制限
    character_list = [CHARACTER_START] + (list(comment) + [CHARACTER_END]*args.max_comment_length)[:args.max_comment_length]
    batch.append((img_path, character_list))
    # バッチ分読み込む
    if len(batch) < args.batch_size:
        continue

    ### image features
    image_features = []
    for img_path, each_comment in batch:
        img_array = xp.array(illust2comment.utility.img2array(illust2comment.utility.load_image(img_path)))
        feature = image_model.feature(img_array, volatile=True)
        image_features.append(feature)
    image_features_concat = chainer.functions.concat(image_features, axis=0)
    image_features_concat = chainer.Variable(image_features_concat.data.copy()) #backwardいらないので。

    ### comment features
    # 最大文字列長分だけforwardする
    comment_model.reset_state()
    predicted = None
    for character_index in xrange(0, args.max_comment_length):
        xs = [
            character_embedder.embed_id(each_comment[character_index])
            for _, each_comment in batch
        ]
        each_predicted = comment_model.feature(
            chainer.Variable(xp.array(xs, dtype=xp.int32)))
        if args.average_model:
            if predicted is None:
                predicted = each_predicted
            predicted += each_predicted
    if not args.average_model: # 最後の出力だけを使う
        predicted = each_predicted

        ### calc loss and update
    loss = chainer.functions.mean_squared_error(
        predicted,
        image_features_concat
    )
    # print("min(comment):", predicted.data.min())
    # print("max(comment):", predicted.data.max())
    # print("min(comment.0):", predicted.data[:, 0].min())
    # print("max(comment.0):", predicted.data[:, 0].max())
    # print("std(comment.0):", predicted.data[:, 0].std())
    # print("min(image):", image_features_concat.data.min())
    # print("max(image):", image_features_concat.data.max())
    # print("std(image.0):", image_features_concat.data[:, 0].std())
    optimizer.zero_grads()
    loss.backward()
    loss.unchain_backward()
    optimizer.update()
    illust2comment.utility.print_ltsv({
        "update": n,
        "loss": loss.data})

    sys.stdout.flush()
    # save model
    print(n)
    if (n/args.batch_size) % 1000 == 0:
        chainer.serializers.save_hdf5(os.path.join(args.model_output_dir, "model_{}".format(n)), comment_model)

    # reset state
    batch = []
