from typing import Any, Dict, Optional, List, Union
import csv
import argparse
import re
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# 手動でアノテーションしたデータをcsv形式で出力 (あんま使わない)
def output_manualy_annotated_results(
            input_data: list,
            output_path:str,
            num_return_sequences=30,
            num_pairs=25,
            white_rels_list=["/r/AtLocation", "/r/IsA", "/r/MadeOf", "/r/PartOf", "/r/Synonym"]
        ):


    with open(output_path, 'w') as wf:
        writer = csv.writer(wf)
        for i in range(num_pairs):
            rel, head, tail = (input_data[(num_return_sequences+1)*i]).split(", ")
            tail = tail.rstrip("\n")
            label = []
            assert rel.startswith("/r")
            if rel not in white_rels_list:
                continue
            else:
                label = [int(input_data[j][0]) for j in range((num_return_sequences+1)*i+1, (num_return_sequences+1)*(i+1))]
                assert len(label) == num_return_sequences
                writer.writerow((rel, head, tail, label))


# 手動評価と自動評価の結果を結合してcsv出力
def merge_m_and_a_results(
        input_data_m: List,
        input_data_a: List,
        output_path: str,
        output_dir: str,
        num_return_sequences=30,
        num_pairs=25):
    """input_data_m
    inhomogenious list
    """

    """input_data_a
    0) rel: str
    1) head: str
    2) tail: str
    3) completions: List[str]
    4) ranks: List[Tuple(int, int)]
    5) rrs: List[Tuple(float, float)]
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w') as wf:
        writer = csv.writer(wf)
        for i in range(num_pairs):
            rel, head, tail = (input_data_m[(num_return_sequences+1)*i]).split(", ")
            tail = tail.rstrip("\n")
            assert head == input_data_a[i][1] and tail == input_data_a[i][2], f"{head} & {input_data_a[i][1]}; {tail} & {input_data_a[i][2]}"
            label_m = [int(input_data_m[j][0]) for j in range((num_return_sequences+1)*i+1, (num_return_sequences+1)*(i+1))]
            assert len(label_m) == num_return_sequences
            # completions_m = [(input_data_m[j][2:]).rstrip("\n") for j in range((num_return_sequences+1)*i+1, (num_return_sequences+1)*(i+1))]
            completions_a = [completion_a for completion_a in eval(input_data_a[i][3])]
            ranks_a = [rank for rank in eval(input_data_a[i][4])]
            rrs_a = [rr for rr in eval(input_data_a[i][5])]
            scores_a = [score for score in eval(input_data_a[i][6])]
            
            # MRRの和を第1キーに，softmax_scoreの和を第2キーにして，全て降順にソートする
            sorted_rrs_indices = sorted(range(num_return_sequences),
                                        key=lambda j: (rrs_a[j][0]+rrs_a[j][1], scores_a[j][0]+scores_a[j][1]),
                                        reverse=True)
            sorted_completions = [completions_a[j] for j in sorted_rrs_indices]
            sorted_ranks = [ranks_a[j] for j in sorted_rrs_indices]
            sorted_rrs = [rrs_a[j] for j in sorted_rrs_indices]
            sorted_scores = [scores_a[j] for j in sorted_rrs_indices]
            sorted_labels_m = [label_m[j] for j in sorted_rrs_indices]

            # 全体の結果を1行に出力
            output_row = [rel, head, tail, sorted_completions, sorted_ranks, sorted_rrs, sorted_scores, sorted_labels_m]
            writer.writerow(output_row)

            # 各completionの結果に関して個別のファイルに出力
            output_path_detail = f"{output_dir}/{rel.lstrip('/r/')}_{head}_{tail}.csv"
            with open(output_path_detail, 'w') as f:
                writer_detail = csv.writer(f)
                writer_detail.writerow((rel, head, tail))
                for cpl, rank, rr, score, label in zip(sorted_completions, sorted_ranks, sorted_rrs, sorted_scores, sorted_labels_m):
                    writer_detail.writerow((cpl, rank, rr, score, label))
                
    print(f"Successfully dumple {output_path}!")


# 手動評価と自動評価の相関を見るために散布図をプロット (あんま役に立たない)
def scatter_diff_between_m_and_a(input_data, output_dir, model_name="Matsuo10"):
    """input_data
    0) relation: str
    1) head: str
    2) tail: str
    3) completions: List[str]
    4) ranks: List[Tuple(int, int)]
    5) rrs: List[Tuple(float, float)]
    6) labels: List[int]
    """

    all_head_rrs_f = []
    all_head_rrs_t = []
    all_tail_rrs_f = []
    all_tail_rrs_t = []
    in_sentence_head_rrs_f = []
    in_sentence_head_rrs_t = []
    in_sentence_tail_rrs_f = []
    in_sentence_tail_rrs_t = []
    for row in input_data:
        for rank_pair, rr_pair, label in zip(eval(row[4]), eval(row[5]), eval(row[-1])):
            label = int(label)
            head_rank, tail_rank = rank_pair
            head_rr, tail_rr = rr_pair
            if label == 1:
                all_head_rrs_t.append(head_rr)
                all_tail_rrs_t.append(tail_rr)
                if head_rank < args.topk*10 and tail_rank < args.topk*10:
                    in_sentence_head_rrs_t.append(head_rr)
                    in_sentence_tail_rrs_t.append(tail_rr)
            else:
                all_head_rrs_f.append(head_rr)
                all_tail_rrs_f.append(tail_rr)
                if head_rank < args.topk*10 and tail_rank < args.topk*10:
                    in_sentence_head_rrs_f.append(head_rr)
                    in_sentence_tail_rrs_f.append(tail_rr)
    print(f"head_f: {len(in_sentence_head_rrs_f)}/{len(all_head_rrs_f)}")
    print(f"tail_f: {len(in_sentence_tail_rrs_f)}/{len(all_tail_rrs_f)}")
    print(f"head_t: {len(in_sentence_head_rrs_t)}/{len(all_head_rrs_t)}")
    print(f"tail_t: {len(in_sentence_tail_rrs_t)}/{len(all_tail_rrs_t)}")

    # 散布図（completion中にheadとtailどちらも含まれる場合）
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.scatter(in_sentence_head_rrs_t, in_sentence_tail_rrs_t, c="green", s=5)
    plt.scatter(in_sentence_head_rrs_f, in_sentence_tail_rrs_f, c="red", s=5)
    plt.xlabel("head")
    plt.ylabel("tail")
    plt.title(f"Diff for In_sentence with {model_name}")
    fig_path = f"{output_dir}/diff_in_sentence.png"
    plt.savefig(fig_path)
    plt.close()

    # 散布図（全部載せ）
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.scatter(all_head_rrs_t, all_tail_rrs_t, c="green", s=5)
    plt.scatter(all_head_rrs_f, all_tail_rrs_f, c="red", s=5)
    plt.xlabel("head")
    plt.ylabel("tail")
    plt.title(f"Diff for All with {model_name}")

    fig_path = f"{output_dir}/diff_all.png"
    plt.savefig(fig_path)
    plt.close()


# 手動評価と自動評価の相関を見るために折れ線グラフをプロット
def plot_line_graphs(input_data: List,
                     output_dir: str,
                     num_return_sequences=30,
                     rel_type_num=5,
                     row_num_per_rel_type=5):
    """input_data
    0) relation: str
    1) head: str
    2) tail: str
    3) completions: List[str]
    4) ranks: List[Tuple(int, int)]
    5) rrs: List[Tuple(float, float)]
    6) softmax_scores: List[Tuple(float, float)]
    7) labels: List[int]
    """

    all_labels = np.array([eval(row[-1]) for row in input_data])
    all_rrs = [eval(row[5]) for row in input_data]
    all_mrrs = np.array([[np.average(pair) for pair in row] for row in all_rrs])
    assert all_labels.shape == tuple((rel_type_num*row_num_per_rel_type, num_return_sequences))
    rels = [row[0] for row in input_data]
    assert len(rels) == rel_type_num*row_num_per_rel_type

    # 折れ線グラフ
    avg_labels = np.average(all_labels, axis=0)
    plt.figure()
    plt.plot(range(len(avg_labels)), avg_labels)
    plt.title("Avg of binary labels sorted by sum of reciprocal rank")
    plt.xlabel("Rank")
    plt.ylabel("Avg of manually labeled score")
    plt.ylim(-0.1,1.1)
    fig_path = f"{output_dir}/line_plot_all.png"
    plt.savefig(fig_path)
    plt.close()

    # 累積折れ線グラフをプロット
    cumulative_labels = np.array([[any(labels[:i+1]) for i in range(len(labels))] for labels in all_labels])
    avg_cumulative_labels = np.average(cumulative_labels, axis=0)
    plt.figure()
    plt.plot(range(1, len(avg_cumulative_labels)+1), avg_cumulative_labels)
    plt.title("Avg of CDF sorted by sum of reciprocal rank")
    plt.xlabel("Rank")
    plt.ylabel("Avg of manually labeled score")
    plt.ylim(-0.1,1.1)
    plt.grid()
    fig_path = f"{output_dir}/line_plot_cdf.png"
    plt.savefig(fig_path)
    plt.close()

    # ついでにcsvも出力
    output_path1 = f"{output_dir}/cdf_all.csv"
    output_path2 = f"{output_dir}/mrr.csv"
    with open(output_path1, 'w') as wf1, open(output_path2, 'w') as wf2:
        writer1 = csv.writer(wf1)
        writer2 = csv.writer(wf2)
        for row, labels, mrr in zip(input_data, cumulative_labels, all_mrrs):
            labels = [int(l) for l in labels]
            writer1.writerow([*row[:3], *labels])
            writer2.writerow([*row[:3], *mrr])

    # x個同時にランダムにラベルを取得した時少なくとも1つ以上当たりが含まれる確率を計算する (xは確率変数)

    # x個同時にとった時，全て外れである確率を求めるための下準備
    probability_spaces = np.array([[(len(row)-sum(row)-i)/(len(row)-i) if len(row)-sum(row)-i>0 else 0 for i in range(len(row))]
            for row in all_labels])

    # x個同時にとった時，全て外れである確率を求める (xは確率変数)
    at_least_prob_spaces = [[1-np.prod(row[:i+1]) for i in range(len(row))] for row in probability_spaces]

    # バッチ方向に平均をとってグラフ出力
    avg_at_least_prob_spaces = np.average(at_least_prob_spaces, axis=0)
    plt.figure()
    plt.plot(range(1, len(avg_at_least_prob_spaces)+1), avg_at_least_prob_spaces)
    plt.title("Avg of prob density func")
    plt.xlabel("Random variable representing the number of simultaneous selections")
    plt.ylabel("Avg of PDF")
    plt.ylim(-0.1,1.1)
    plt.grid()
    fig_path = f"{output_dir}/line_plot_avg_of_pdf.png"
    plt.savefig(fig_path)
    plt.close()

    # ついでにcsvも出力
    output_path = f"{output_dir}/avg_of_pdf.csv"
    with open(output_path, 'w') as wf:
        writer = csv.writer(wf)
        for row, pdf in zip(input_data, at_least_prob_spaces):
            writer.writerow([*row[:3], *pdf])

    # ランダム選択と提案手法の比較
    plt.figure()
    plt.plot(range(1, len(avg_at_least_prob_spaces)+1), avg_at_least_prob_spaces, c="blue", label="Random")
    plt.plot(range(1, len(avg_cumulative_labels)+1), avg_cumulative_labels, c="orange", label="Ours")
    plt.title("Diffrence between random selection and ours")
    plt.xlabel("Rank")
    plt.ylabel("Prob")
    plt.legend(loc="lower right")
    plt.xlim()
    plt.ylim(-0.05,1.1)
    plt.grid()
    fig_path = f"{output_dir}/line_plot_difference.png"
    plt.savefig(fig_path)
    plt.close()

    print("All done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="Matsuo10")
    parser.add_argument('--num_pairs', type=int, default=25)
    parser.add_argument('--num_return_sequences', type=int, default=30)
    parser.add_argument('--seed', type=int, default=19990429)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    result_dir = args.result_dir

    input_path_m = f"{result_dir}/formatted_results_.txt"
    input_path_a = f"{result_dir}/rated_results.csv"
    with open(input_path_m, 'r') as mf, open(input_path_a, 'r') as af:
        input_data_m = [line for line in mf if line.strip()]
        assert len(input_data_m) == (args.num_return_sequences + 1) * args.num_pairs
        a_reader = csv.reader(af)
        input_data_a = [line for line in a_reader]
    output_path = f"{result_dir}/diffs_btween_manda.csv"
    output_dir = f"{result_dir}/diffs_btween_manda"
    merge_m_and_a_results(input_data_m, input_data_a, output_path, output_dir)

    input_path = f"{result_dir}/diffs_btween_manda.csv"
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        input_data = [row for row in reader]
    """input_data
    0) relation: str
    1) head: str
    2) tail: str
    3) completions: List[str]
    4) ranks: List[Tuple(int, int)]
    5) rrs: List[Tuple(float, float)]
    6) labels: List[int]
    """
    # 手動でラベル付した結果をエクセルにコピペしやすいように`labels.txt`を出力
    output_path = f"{result_dir}/labels.txt"
    with open(output_path, 'w') as wf:
        labels = [eval(row[-1]) for row in input_data]
        for row in labels:
            label_sum = sum(row)
            wf.write(f"{label_sum}\n")
    
    num_output_cpls = args.topk_cpls
    output_path = f"{result_dir}/diffs_btween_manda_formatted{num_output_cpls}.txt"
    tmp_input_data = [[*row[:3],
                       (eval(row[3]))[:num_output_cpls],
                       (eval(row[4]))[:num_output_cpls],
                       (eval(row[5]))[:num_output_cpls],
                       (eval(row[6]))[:num_output_cpls],
                       (eval(row[-1]))[:num_output_cpls],
                       ]for row in input_data]
    tmp_input_data.sort(key=lambda x: np.average(x[5]), reverse=True)
    with open(output_path, 'w') as wf:
        for row in tmp_input_data:
            wf.write(f"{row[0]} {row[1]} {row[2]}\n")
            rrs_list = []
            label_sum = 0
            for i, (cpl, rrs, scores, label) in enumerate(zip(row[3], row[5], row[6], row[-1])):
                rrs_str = f"({rrs[0]:.2f},{rrs[1]:.2f})"
                scores_str = f"({scores[0]:.2f},{scores[1]:.2f})"
                if i >= num_output_cpls:
                    break
                else:
                    wf.write(f"{label},{rrs_str},{scores_str},{cpl}\n")
                    rrs_list.append(rrs)
                    label_sum += int(label)
            rrs_np_list = np.array(rrs_list)
            wf.write(f"sum:{label_sum}, head:{np.average(rrs_np_list[:,0]):.2f}, tail:{np.average(rrs_np_list[:,1]):.2f}, all:{np.average(rrs_np_list):.2f}\n")
            wf.write("\n")

    scatter_diff_between_m_and_a(input_data, result_dir)
    plot_line_graphs(input_data, result_dir)
