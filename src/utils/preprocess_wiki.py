import gzip
from tqdm import tqdm
import datetime
import sys

from normalize_neologd import normalize_neologd, subtract_hashtag


corpus_dir = "datasets/jawiki-20221226"
input_corpus_path = f"{corpus_dir}/jawiki_origin.txt.gz"
output_corpus_path = f"{corpus_dir}/jawiki_sentences.txt.gz"

print(f"Loading {input_corpus_path}")
with gzip.open(input_corpus_path, "rt") as f:
    wiki_corpus = f.readlines()

print(f"Dumping {output_corpus_path}")
sys.stdout.flush()
with gzip.open(output_corpus_path, "wt") as f:
    i = 0
    for article in wiki_corpus:
        i += 1
        if i % 10000 == 0:
            print(f"{datetime.datetime.now()}: {i} articles has been processed.")
            sys.stdout.flush()
        sentence_list = subtract_hashtag(normalize_neologd(article)).split("。")
        f.write("\n".join(sentence_list))

print(f"Successfully dumped {output_corpus_path}")

limit = 1000
output_1000_corpus_path = f"{corpus_dir}/jawiki_sentences_1000.txt.gz"
with gzip.open(output_corpus_path, "rt") as rf:
    with gzip.open(output_1000_corpus_path, "wt") as wf:
        for line in rf:
            wf.write(line[:limit])
            if len(line) > limit:
                wf.write('\n')
print("All done !")
