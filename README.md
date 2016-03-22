ニコニコ超コメント生成コンテスト サンプルプログラム
=====================================================

画像 -> コメント検索の試し方
------------------------------------

A. 依存パッケージのインストール

```
pip install -r requirements.txt
```

B. 必要なファイルをダウンロードする

ダウンロード用signitureはニコニコ超コメント生成コンテストの登録ページから別途申請して手に入れてください。

```bash
signature="XXXXX"
mkdir data
wget "https://nico-opendata.jp/comment-hackathon/nico_comment_feature_v1.hdf5$signiture" -O ./data/nico_comment_feature_v1.hdf5
wget "https://nico-opendata.jp/comment-hackathon/nico_comment_feature_v1_vocabulary.txt$signiture" -O ./data/nico_comment_feature_v1_vocabulary.txt
wget "https://nico-opendata.jp/comment-hackathon/nico_illust_tag_v2.hdf5$signiture" -O ./data/nico_illust_tag_v2.hdf5
wget "https://nico-opendata.jp/comment-hackathon/seiga_comment.tsv$signiture" -O ./data/seiga_comment.tsv
```

C. 学習しやすいコメントに絞って、ランダムにソートする

```bash
python prettify_comment.py --max_comment_count=100 --min_comment_length=4 --max_comment_length=30 data/seiga_comment.tsv > data/seiga_comment_random.tsv
```

D. ipythonを実行

```bash
ipython notebook search_comment_example.ipynb
```

類似画像検索の試し方
------------------------------------

A, B, Cは上に同じ

D. ipythonを実行

```bash
ipython notebook search_comment_example.ipynb
```
