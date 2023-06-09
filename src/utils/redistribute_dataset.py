import gzip
import csv
from sklearn.model_selection import train_test_split
import random
import sys


# 参照. datasets/conceptnet-assertions-5.7.0/ja/relations.csv
relations_data_path = "datasets/conceptnet-assertions-5.7.0/ja/relations.csv"
with open(relations_data_path, "r") as f:
    reader = csv.reader(f)
    all_data = {}
    for row in reader:
        all_data[row[2]] = []

input_dir = "datasets/rel_gen/cleaned_rhts"
old_paths = [f"{input_dir}/cleaned_rhts_200_1.csv.gz",
                   f"{input_dir}/cleaned_rhts_200_2.csv.gz",
                   f"{input_dir}/cleaned_rhts_200_3.csv.gz",
                   f"{input_dir}/cleaned_rhts_200_4.csv.gz",
                   ]

new_train_data = []
new_val_data = []
new_test_data = []

for old_path in old_paths:
    with gzip.open(old_path, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == "[]":
                continue
            elif row[0] == "関連する":
                new_test_data.append(row)
            else:
                all_data[row[0]].append(row)

for relation, data in all_data.items():
    if len(data) < 1:
            continue
    train, val_and_test = train_test_split(data,
                                           train_size=0.8,
                                           test_size=0.2,
                                           shuffle=True,
                                           random_state=19990429)
    if len(val_and_test) < 2:
            continue
    val, test = train_test_split(val_and_test,
                                 train_size=0.5,
                                 test_size=0.5,
                                 shuffle=True,
                                 random_state=19990429)
    new_train_data.extend(train)
    new_val_data.extend(val)
    new_test_data.extend(test)

random.shuffle(new_train_data)
random.shuffle(new_val_data)
random.shuffle(new_test_data)

output_dir = "datasets/rel_gen/redistributed_rhts"
new_train_path = f"{output_dir}/rhts_200_train.csv.gz"
new_val_path = f"{output_dir}/rhts_200_val.csv.gz"
new_test_path = f"{output_dir}/rhts_200_test.csv.gz"

with gzip.open(new_train_path, "wt") as f:
    writer = csv.writer(f)
    writer.writerows(new_train_data)

with gzip.open(new_val_path, "wt") as f:
    writer = csv.writer(f)
    writer.writerows(new_val_data)

with gzip.open(new_test_path, "wt") as f:
    writer = csv.writer(f)
    writer.writerows(new_test_data)

