import gzip
from tqdm import tqdm
import datetime
import sys
import re

from normalize_neologd import normalize_neologd, subtract_hashtag


corpus_dir = "datasets/jawiki-20221226"
input_corpus_path = f"{corpus_dir}/jawiki_origin.txt.gz"
output_corpus_path = f"{corpus_dir}/jawiki_sentences.txt.gz"

print(f"Loading {input_corpus_path}")
with gzip.open(input_corpus_path, "rt") as f:
    wiki_corpus = f.readlines()

# クォーテーションマークで囲まれた部分を抽出する正規表現パターン
pattern = re.compile(r'「.+?」')

print(f"Dumping {output_corpus_path}")
sys.stdout.flush()
with gzip.open(output_corpus_path, "wt") as f:
    for i, article in enumerate(wiki_corpus):
        if i % 10000 == 0:
            print(f"{datetime.datetime.now()}: {i} articles has been processed.")
            sys.stdout.flush()

        # 正規化
        normalized_article = subtract_hashtag(normalize_neologd(article))

        # クォーテーションマークで囲まれた部分を一時的に置換しておく
        replaced_article = pattern.sub(lambda x: x.group().replace("。", "＠＠＠"), normalized_article)

        # 「。」で改行する
        sentences = replaced_article.split("。")
        formatted_text = "\n".join(sentence.replace("＠＠＠", "。").strip() for sentence in sentences if sentence != "")

        # 文末の「。」を削除する
        formatted_text = formatted_text.rstrip("。")

        # クォーテーションマークで囲まれた部分を元に戻す
        sentence_list = pattern.sub(lambda x: x.group().replace("＠＠＠", "。"), formatted_text)
        f.write(sentence_list)

print(f"Successfully dumped {output_corpus_path}")

limit = 200
output_1000_corpus_path = f"{corpus_dir}/jawiki_sentences_200.txt.gz"
with gzip.open(output_corpus_path, "rt") as rf:
    with gzip.open(output_1000_corpus_path, "wt") as wf:
        for line in rf:
            wf.write(line[:limit])
            if len(line) > limit:
                wf.write('\n')
print("All done !")
