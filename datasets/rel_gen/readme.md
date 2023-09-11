# rel_gen
ConceptNetのトリプレットに対して日本語wikipediaから文を抽出してきた結果に関するファイル群

## origin_rhts/origin_rhts_200_[i].csv.gz

### 概要
- `datasets/conceptnet-assertions-5.7.0/ja/triplets_[i].csv`のheadとtailに対し、`src/utils/extract_sentences.py`の`extract_sentences`で`datasets/jawiki-20221226/jawiki_sentences_200.txt.gz`からheadとtailを含む文を全て抽出してきたもの
- relation, head, tail, sentencesの4列からなる
- 抽出文の最大長は200だが、改行コードが含まれてしまっている
- また、文を抽出できなかった組も含まれている

### 内容
```
$ zcat 'datasets/rel_gen/origin_rhts/origin_rhts_200_1.csv.gz' | head -n 1
から派生した,衣虱,衣,"['ヒトに寄生するヒトジラミ(人虱、Pediculuswikt:en:humanus)は2つの亜種、主に毛髪に寄宿するアタマジラミ(頭虱、P.h.capitis)と、主に衣服に寄宿するコロモジラミ(衣虱、P.h.wikt:en:humanus,P.wikt:en:humanuscorporis)に分けられる\n']"
```

## cleaned_rhts/cleaned_rhts_200_[i].csv.gz
- `origin_rhts/origin_rhts_200_[i].csv.gz`に対し、`src/utils/extract_sentences.py`の`clean_sentences`で抽出文のない組を削除し、抽出文の文末にある改行コードを除去したもの
- relation, head, tail, sentencesの4列からなる
- 1行あたり平均3388.01文抽出
- 各行数については以下の通り (合計: 204,027/250,541)
    - cleaned_rhts_200_1.csv.gz: 51,050
    - cleaned_rhts_200_2.csv.gz: 51,003
    - cleaned_rhts_200_3.csv.gz: 50,965
    - cleaned_rhts_200_4.csv.gz: 50,898
    - cleaned_rhts_200_5.csv.gz: 28
    - cleaned_rhts_200_6.csv.gz: 34
    - cleaned_rhts_200_7.csv.gz: 28
    - cleaned_rhts_200_8.csv.gz: 21
    - cleaned_rhts_200_5678.csv.gz: 111 (5から8を統合したもの)

- `*_t.csv.gz`は抽出文が1,000,000未満であるもの, `*_f.csv.gz`は1,000,000以上であるもの
    - cleaned_rhts_200_1_over1m.csv.gz: 23 (5.3G)
    - cleaned_rhts_200_1_t.csv.gz: 51,027 (8.0G)
    - cleaned_rhts_200_2_over1m.csv.gz: 13 (3.5G)
    - cleaned_rhts_200_2_t.csv.gz: 50,990 (7.1G)
    - cleaned_rhts_200_3_over1m.csv.gz: 32 (8.5G)
    - cleaned_rhts_200_3_t.csv.gz: 50,933 (7.0G)
    - cleaned_rhts_200_4_over1m.csv.gz: 23 (5.2G)
    - cleaned_rhts_200_4_t.csv.gz: 50,875 (7.4G)

## cleaned_rhts/cleaned_htns_200.csv.gz
- 全ての`*_t.csv.gz`に対して、語の組み合わせの重複を許さないようにしたもの
- head, tail, len(sentences), sentences の4列からなる（headとtailは順不同）
- 184,612行ある
- 23GB

## cleaned_rhts/cleaned_htns_200_over200k.csv.gz
- `cleaned_htns_200.csv.gz`のうち抽出文が200,000を超えるもの
- head, tail, len(sentences), sentences の4列からなる（headとtailは順不同）
- 226行ある
- 7.1GB

## cleaned_rhts/cleaned_htns_200_[i]_ascending.csv.gz
- `cleaned_htns_200.csv.gz`のうち抽出文が200,000未満のものを等分したもの
- 抽出文の昇順にソートしてある
- 語彙サイズ: 68,848
- cleaned_htns_200_[i]_ascending_1.csv.gzは消さないように。1以外は消して良い。
- 全部で184,385行 (15.3G)
    - cleaned_htns_200_1_ascending_1.csv.gz: 46,097 (3.7G)
    - cleaned_htns_200_2_ascending_1.csv.gz: 46,096 (4.0G)
    - cleaned_htns_200_3_ascending_1.csv.gz: 46,096 (3.7G)
    - cleaned_htns_200_4_ascending_1.csv.gz: 46,096 (3.8G)


## redistributed_rhts/*
- `cleaned_rhts/cleaned_rhts_200_[i].csv.gz`全てのファイルを統合して、`src/utils/redistribute_dataset.py`でtrain/validation/testに3分割したもの
- `関連がある（/r/RelatedTo）`を全てtestに回し、それ以外は8/1/1で分割


## statistics/*
`dataset_statistics.ipynb`で調べた各種統計情報が入っている
