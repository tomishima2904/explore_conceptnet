SEED=1
python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py' \
        --device_type=cuda:1 \
        --model=matsuo-lab/weblab-10b \
        --num_return_sequences=30 \
        --num_refs=0 \
        --template_name=zero-shot_no_refs_5 \
        --seed=$SEED \
        --input_path="datasets/連想語頻度表/all/dev_mini2/htrkpns_tmp.csv.gz"

# rinna/japanese-gpt-neox-3.6b
# cyberagent/open-calm-7b
# matsuo-lab/weblab-10b
# pfnet/plamo-13b
