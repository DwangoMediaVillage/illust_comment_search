# -*- coding: utf-8 -*-
import numpy
import codecs
from PIL import Image

def img2array(img):
    data_subtracted = numpy.asarray(img)[:,:,:3].astype(numpy.float32) - 128
    data = data_subtracted.transpose(2, 0, 1)[::-1]
    return numpy.array([data])

def load_image(img_file):
    img = Image.open(img_file).resize((224,224))
    if len(img.size) == 2: #gray scale
        img_rgb = Image.new("RGB", img.size)
        img_rgb.paste(img)
        img = img_rgb
    return img

def load_id_comments(comment_file, easy_comment=False):
    while True: #何エポックでも読み込む
        for line in codecs.open(comment_file, 'r', 'utf-8'):
            values = line.rstrip().split("\t")
            if not len(values) == 2:
                continue
            content_id, comment = line.rstrip().split("\t")
            yield content_id, comment

def print_ltsv(raw_dict):
    items = []
    for key, value in raw_dict.items():
        items.append("{}:{}".format(key,value))
    print("\t".join(items))

