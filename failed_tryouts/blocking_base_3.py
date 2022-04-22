from collections import defaultdict

from pandas import DataFrame
from tqdm import tqdm
import pandas as pd
import re
import time


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


def compute_recall(X, Y):
    c = 0
    st = set(X)
    for i in range(Y.shape[0]):
        t = (Y['lid'][i], Y['rid'][i])
        if t in st:
            c += 1
        # else:
        #     print(t)
    return c / len(Y)


def block_X2(X: DataFrame):
    X = X.sort_values(by='name', key=lambda x: x.str[:5])
    jeccard_tolerance = 10
    jeccard_threshold = 0.5
    candidate_pairs_real_ids = []
    jaccard_similarities = []
    for j in range(X.shape[0]):
        jeccard_passed = 0
        for k in range(j + 1, X.shape[0]):
            name1 = str(X['name'][j])
            name2 = str(X['name'][k])
            s1 = set(name1.lower().split(" "))
            s2 = set(name2.lower().split(" "))
            s = len(s1.intersection(s2)) / max(len(s1), len(s2))
            if jeccard_passed >= jeccard_tolerance:
                break
            if s < jeccard_threshold:
                jeccard_passed += 1

            jaccard_similarities.append(s)
            id1, id2 = X['id'][j], X['id'][k]
            if id2 > id1:
                candidate_pairs_real_ids.append((id1, id2))
            else:
                candidate_pairs_real_ids.append((id2, id1))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


def block_X1(X):
    # build index from patterns to tuples
    pattern2id = [defaultdict(list) for _ in range(5)]
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X["title"][i])
        pattern_2 = re.findall("\w+\s\w+\d+", attr_i)  # look for patterns like "thinkpad x1"
        if len(pattern_2) != 0:
            pattern_2 = list(sorted(pattern_2))
            pattern_2 = [str(it).lower() for it in pattern_2]
            pattern2id[1]["".join(pattern_2)].append(i)
        pattern_3 = "".join([str(it).lower() for it in attr_i.split(" ")[:3]])
        pattern2id[2][pattern_3].append(i)
        pattern_4 = sum([int(d) for d in re.findall(r"\d+", attr_i)])
        pattern2id[3][pattern_4].append(i)
        pattern_5 = "".join([str(it).lower() for it in attr_i.split(" ")[-3:]])
        pattern2id[4][pattern_5].append(i)
    # add id pairs that share the same pattern to candidate set
    candidate_pairs_real_ids = []
    jaccard_similarities = []
    ii = 0
    for p2id in pattern2id:
        ii += 1
        for pattern in tqdm(p2id):
            ids = sorted(p2id[pattern])
            if len(ids) < 150:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        real_id1 = X['id'][ids[i]]
                        real_id2 = X['id'][ids[j]]
                        # candidate_pairs.add((ids[i], ids[j]))
                        if real_id1 < real_id2:
                            candidate_pairs_real_ids.append((real_id1, real_id2))
                        else:
                            candidate_pairs_real_ids.append((real_id2, real_id1))
                        # compute jaccard similarity
                        name1 = str(X['title'][ids[i]])
                        name2 = str(X['title'][ids[j]])
                        s1 = set(name1.lower().split())
                        s2 = set(name2.lower().split())
                        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


if __name__ == "__main__":
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("../X1.csv")
    X2 = pd.read_csv("../X2.csv")

    # perform blocking
    X1_candidate_pairs = block_X1(X1)
    X2_candidate_pairs = block_X2(X2)
    print(len(X1_candidate_pairs))
    print(len(X2_candidate_pairs))
    print(f'CANDIDATE PAIRS X1 :{len(X1_candidate_pairs)}')
    print(f'CANDIDATE PAIRS X2 :{len(X2_candidate_pairs)}')
    # save results
    # save_output(X1_candidate_pairs, X2_candidate_pairs)
    # print(time.time() - s)
    rc1 = compute_recall(X1_candidate_pairs, pd.read_csv("../Y1.csv"))
    print(f'X1 Recall: {rc1}')
    rc2 = compute_recall(X2_candidate_pairs, pd.read_csv("../Y2.csv"))
    print(f'X2 Recall: {rc2}')
    print(f'Average Recall: {rc1 * 1 / 3 + rc2 * 2 / 3}')
