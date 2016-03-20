seiga_comment_example
=====================

公開用静画コメント生成


画像 -> コメント検索の試し方
==================================

1. 必要なファイルをダウンロードする

ダウンロードURLは別途申請して手に入れてください。

```bash
mkdir data
wget https://nico-opendata.jp/******/nico_comment_feature_v1.hdf5 ./data
wget https://nico-opendata.jp/******/nico_comment_feature_v1_vocabulary.txt ./data
wget https://nico-opendata.jp/******/nico_illust_tag_v2.hdf5 ./data
wget https://nico-opendata.jp/******/seiga_comment.tsv ./data
```

2. 学習しやすいコメントに絞って、ランダムにソートする

```bash
python prettify_comment.py --max_comment_count=100 --min_comment_length=4 --max_comment_length=30 data/seiga_comment.tsv > data/seiga_comment_random.tsv
```

3. ipythonを実行

```bash
ipython notebook search_comment_example.ipynb
```
