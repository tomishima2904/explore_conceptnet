SEED=1

# 連想理由生成を行った後，提案手法によるスコアリングをして選定を行う

python '/work/tomishima2904/explore_conceptnet/src/models/gen_with_mlm_selection.py'  \
        --device_type=cuda:0 \
        --dir_name="evaluation/master_alpha" \
        --model=matsuo-lab/weblab-10b \
        --num_pre_return_sequences=30 \
        --num_return_sequences=3 \
        --num_refs=0 \
        --template_dir=datasets/連想語頻度表/templates/master30 \
        --template_name="5-shot_len_dedupulicated" \
        --seed=$SEED

# template_name
# - 5-shot_len_deduplicated: 提案手法
# - 5-shot_len_deduplicated_random: 自動ランダム
# - 5-shot_len_mR: 手動1 (富島)
# - 5-shot_len_mT: 手動2 (問井)
# - 5-shot_len_mY: 手動3 (悠真)
# 実験に使うtemplateは `datasets/連想語頻度表/templates/master30` にある
# 上記の `{template_name}.json` が `template_dir` に保存されている

# 生成は 各種パラメータを定めたら，`./do_3.sh` をするだけ（全部終わるのに30分くらいかかるかも）

# 出力結果は `results/ja/連想語頻度表/{dir_name}` に保存
# formmated_results.txt で二値評価を行ったら，必ずそのファイル名を `formmated_results_.txt` と変える
# 名称変更した後，`src/utils/result_extractor.ipynb` の `Labels.txt` という項目で，
# `result_dir` を実験結果が保存されているディレクトリ名に指定して，セルを実行すれば `labels.txt` が出力される
# `label.txt` は `formmated_results_.txt` の結果をエクセル等に貼りやすく集計した結果が保存されている
# `labels.txt` は 刺激語と連想語の組の数の長さの列ベクトルが出力されるので，エクセル等に貼り付けて使える

