import re
import time
from difflib import get_close_matches

import pandas as pd
import spacy
from pandarallel import pandarallel
from pandas import DataFrame

pandarallel.initialize(progress_bar=True, use_memory_fs=False)

ner = spacy.load("../ner_model_0", disable=['tagger', 'parser'])
stopwords = ner.Defaults.stop_words


def apply_ner(text):
    entities = ner(text)
    entities_text = set()
    for entity in entities:
        lower = entity.text.lower()
        # if lower not in stopwords and not re.match(r'[^A-Za-z]', lower):
        #     # if lower in entities_text:
        #     #     for i in range(3):
        #     #         new_e = lower+f"_{i}"
        #     #         if new_e not in entities_text:
        #     #             entities_text.add(new_e)
        #     #             break
        #     # else:
        entities_text.add(lower)
    return sorted(entities_text)


def jeccard(s1: set, s2: set):
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 0
    return intersection / union


def common_prefix_length(short: list, long: list):
    c = 0
    for i in range(len(short)):
        if short[i] == long[i]:
            c += 1
        else:
            break
    return c


def sort_from_mid(x: []):
    mid = x[len(x) // 2]


def block_with_attr_eval(X: DataFrame, attr, jeccard_threshold, jeccard_tolerance):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    # NERs are sorted in ASC order
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    # sort NERs in DESC order
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    X['NER_TEXT_MID'] = X['NER'].parallel_apply(
        lambda x: ' '.join(get_close_matches(x[len(x) // 2], x, cutoff=0, n=len(x))))
    # X['NER_SIZE'] = X['NER'].parallel_apply(len)
    candidate_pairs_real_ids = set()
    s = time.time()
    for op in range(3):
        if op == 0:
            X = X.sort_values(by=['NER_TEXT'])
        if op == 1:
            X = X.sort_values(by=['NER_TEXT_REV'])
        if op == 2:
            X = X.sort_values(by=['NER_TEXT_MID'])
        # if op == 2:
        #     X = X.sort_values(by=['NER_SIZE'])
        block = X['id'].values
        block_ner = [set(n) for n in X['NER'].values]
        for j in range(len(block)):
            jeccard_passed = 0
            for k in range(j + 1, len(block)):
                if jeccard_passed > jeccard_tolerance:
                    break
                similarity = jeccard(block_ner[j], block_ner[k])
                if similarity <= jeccard_threshold:
                    jeccard_passed += 1
                    continue
                id1, id2 = block[j], block[k]
                if id1 == 640306 and id2 == 1041786:
                    print(f'JECCARD :{similarity} {id2}')
                if id2 > id1:
                    candidate_pairs_real_ids.add(((id1, id2), similarity))
                else:
                    candidate_pairs_real_ids.add(((id2, id1), similarity))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return candidate_pairs_real_ids


def block_with_attr_X2(X: DataFrame,  jeccard_threshold, jeccard_tolerance):
    s = time.time()
    X['NER'] = X['name'].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    # NERs are sorted in ASC order
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    # sort NERs in DESC order
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    X['NER_TEXT_MID'] = X['NER'].parallel_apply(
        lambda x: ' '.join(get_close_matches(x[len(x) // 2], x, cutoff=0, n=len(x))))
    # X['NER_SIZE'] = X['NER'].parallel_apply(len)
    candidate_pairs_real_ids = set()
    s = time.time()
    for op in range(3):
        if op == 0:
            X = X.sort_values(by=['NER_TEXT'])
        if op == 1:
            X = X.sort_values(by=['NER_TEXT_REV'])
        if op == 2:
            X = X.sort_values(by=['NER_TEXT_MID'])
        # if op == 2:
        #     X = X.sort_values(by=['NER_SIZE'])
        block = X['id'].values
        block_ner = [set(n) for n in X['NER'].values]
        for j in range(len(block)):
            jeccard_passed = 0
            for k in range(j + 1, len(block)):
                if jeccard_passed > jeccard_tolerance:
                    break
                similarity = jeccard(block_ner[j], block_ner[k])
                if similarity < jeccard_threshold:
                    jeccard_passed += 1
                    continue
                id1, id2 = block[j], block[k]
                if id2 > id1:
                    candidate_pairs_real_ids.add(((id1, id2), similarity))
                else:
                    candidate_pairs_real_ids.add(((id2, id1), similarity))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]


def block_with_attr(X: DataFrame, attr, jeccard_threshold, jeccard_tolerance):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    # NERs are sorted in ASC order
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    # sort NERs in DESC order
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    X['NER_TEXT_MID'] = X['NER'].parallel_apply(
        lambda x: ' '.join(get_close_matches(x[len(x) // 2], x, cutoff=0, n=len(x))))
    # X['NER_SIZE'] = X['NER'].parallel_apply(len)
    candidate_pairs_real_ids = set()
    s = time.time()
    for op in range(3):
        if op == 0:
            X = X.sort_values(by=['NER_TEXT'])
        if op == 1:
            X = X.sort_values(by=['NER_TEXT_REV'])
        if op == 2:
            X = X.sort_values(by=['NER_TEXT_MID'])
        # if op == 2:
        #     X = X.sort_values(by=['NER_SIZE'])
        block = X['id'].values
        block_ner = [set(n) for n in X['NER'].values]
        for j in range(len(block)):
            jeccard_passed = 0
            for k in range(j + 1, len(block)):
                if jeccard_passed > jeccard_tolerance:
                    break
                similarity = jeccard(block_ner[j], block_ner[k])
                if similarity < jeccard_threshold:
                    jeccard_passed += 1
                    continue
                id1, id2 = block[j], block[k]
                if id2 > id1:
                    candidate_pairs_real_ids.add(((id1, id2), similarity))
                else:
                    candidate_pairs_real_ids.add(((id2, id1), similarity))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]


def block_with_attr_2(X: DataFrame, attr, common_entities_lim, jeccard_tolerance):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    # NERs are sorted in ASC order
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    # sort NERs in DESC order
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    candidate_pairs_real_ids = set()
    s = time.time()
    for op in range(2):
        if op == 0:
            X = X.sort_values(by=['NER_TEXT'])
        if op == 1:
            X = X.sort_values(by=['NER_TEXT_REV'])
        block = X['id'].values
        block_ner = [set(n) for n in X['NER'].values]
        for j in range(len(block)):
            jeccard_passed = 0
            for k in range(j + 1, len(block)):
                if jeccard_passed > jeccard_tolerance:
                    break
                intersection = len(block_ner[j].intersection(block_ner[k]))
                if intersection < common_entities_lim:
                    jeccard_passed += 1
                    continue
                id1, id2 = block[j], block[k]
                if id2 > id1:
                    candidate_pairs_real_ids.add(((id1, id2), intersection))
                else:
                    candidate_pairs_real_ids.add(((id2, id1), intersection))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]


def save_output(X1_candidate_pairs, X2_candidate_pairs):
    # save the candset for both datasets to a SINGLE file output.csv
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


def compute_precision(X, Y):
    c = 0
    st = {tuple(i) for i in Y.to_numpy()}
    for pair in X:
        if pair in st:
            c += 1
        # else:
        #     print(pair)
    return c / len(X)


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


def eval():
    X1 = pd.read_csv("../X1.csv")
    Y1 = pd.read_csv("../Y1.csv")
    res = block_with_attr_eval(X1, 'title', .5, 10)
    ranges = [i / 10 for i in range(0, 11)]
    buckets = [list() for _ in range(11)]
    for val in res:
        for r in range(len(ranges) - 1):
            if ranges[r] < val[1] <= ranges[r + 1]:
                buckets[r].append(val)
    for r, b in zip(ranges, buckets):
        vals = [v[0] for v in b]
        if len(b) > 0:
            print(r, len(b), compute_recall(vals, Y1), compute_precision(vals, Y1))
    vals = [v[0] for v in res]
    print(f'TOTAL RECALL : {compute_recall(vals, Y1)} TOTAL PRECISION: {compute_precision(vals, Y1)}')
    # r = [v[0] for v in res if v[0][0] == 1041786 or v[0][1] == 1041786]
    # print(len(r))
    # Y1 = Y1.loc[(Y1["lid"] == 1041786) | (Y1["rid"] == 1041786)]
    # print(f'TOTAL RECALL : {compute_recall(r, Y1.reset_index())} TOTAL PRECISION: {compute_precision(r, Y1)}')


def run():
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("../X1.csv")
    X2 = pd.read_csv("../X2.csv")

    # # perform blocking
    #  ORACLES :::
    # X1_candidate_pairs = block_with_attr(X1, "title", 0.06, 192)
    # X2_candidate_pairs = block_with_attr(X2, "name", 0, 1)
    X1_candidate_pairs = block_with_attr(X1, "title", .4, 5)
    X2_candidate_pairs = block_with_attr(X2, "name", .3, 100)
    #
    # # save results
    # save_output(X1_candidate_pairs, X2_candidate_pairs)
    e = time.time()
    print(f'CANDIDATE PAIRS X1 :{len(X1_candidate_pairs)}')
    print(f'CANDIDATE PAIRS X2 :{len(X2_candidate_pairs)}')
    print(f'TOTAL TIME : {e - s}')
    Y1 = pd.read_csv("../Y1.csv")
    Y2 = pd.read_csv("../Y2.csv")
    print(
        f'X1 Recall: {round(compute_recall(X1_candidate_pairs, Y1), 3)} PRECISION: {round(compute_precision(X1_candidate_pairs, Y1), 3)}')
    print(
        f'X2 Recall: {round(compute_recall(X2_candidate_pairs, Y2), 3)} PRECISION: {round(compute_precision(X2_candidate_pairs, Y2), 3)}')


if __name__ == "__main__":
    run()
