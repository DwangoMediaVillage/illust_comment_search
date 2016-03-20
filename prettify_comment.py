# -*- coding: utf-8 -*-
import argparse
import collections
import random
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("comment_tsv")
parser.add_argument("--max_comment_count", type=int, default=100)
parser.add_argument("--min_comment_length", type=int, default=4)
parser.add_argument("--max_comment_length", type=int, default=30)
args = parser.parse_args()

lines = list(codecs.open(args.comment_tsv, 'r', 'utf-8'))
random.shuffle(lines)
comment_count = collections.defaultdict(int) #comments per image
for line in lines:
    values = line.rstrip().split("\t")
    if not len(values) == 2:
        continue
    content_id, comment = values
    if comment_count[content_id] > args.max_comment_count:
        continue
    if args.min_comment_length > len(comment):
        continue
    if len(comment) > args.max_comment_length:
        continue
    if comment[0] == "|" or comment[0] == u"｜": #|こういうの|
        continue
    if comment[0] == u"↑": #返信
        continue
    print(line.rstrip().encode("utf-8"))
    comment_count[content_id] += 1
