import re
import time

import spacy
import pandas as pd

from pandarallel import pandarallel

from pandas import DataFrame

pandarallel.initialize(progress_bar=True)

ner = spacy.load("ner_model_0", disable=['tagger', 'parser'])
stopwords = ner.Defaults.stop_words


def block_with_attr_concept(X: DataFrame, attr):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    window_slide = 50
    X = X.sort_values(by=['NER'])
    X.to_csv('t.csv')
    print(f'NER TIME : {time.time() - s}')
    window_size = 300
    candidate_pairs_real_ids = set()
    s = time.time()
    for i in range(0, X.shape[0], window_slide):
        block = sorted(X['id'][i:i + window_size].values)
        for j in range(len(block)):
            for k in range(j + 1, len(block)):
                candidate_pairs_real_ids.add((block[j], block[k]))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    return list(candidate_pairs_real_ids)


def apply_ner(text):
    entities = ner(text)
    entities_text = set()
    for entity in entities:
        lower = entity.text.lower()
        if lower not in stopwords and not re.match(r'[^A-Za-z]', lower):
            entities_text.add(lower)
    return sorted(entities_text)


def jeccard(s1: set, s2: set):
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union




def block_with_attr_concept_2(X: DataFrame, attr):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    window_slide = 20
    window_size = 100
    candidate_pairs_real_ids = set()
    jeccard_threshold = 0.5
    s = time.time()
    for op in range(2):
        if op == 0:
            X = X.sort_values(by=['NER_TEXT'])
        if op == 1:
            X = X.sort_values(by=['NER_TEXT_REV'])
        for i in range(0, X.shape[0], window_slide):
            relevant_rows = X.iloc[i:i+window_size]
            block = relevant_rows['id'].values
            block_ner = [set(n) for n in relevant_rows['NER'].values]
            for j in range(len(block)):
                for k in range(j + 1, len(block)):
                    if jeccard(block_ner[j], block_ner[k]) < jeccard_threshold:
                        break
                    id1, id2 = block[j], block[k]
                    if id2 > id1:
                        candidate_pairs_real_ids.add((id1, id2))
                    else:
                        candidate_pairs_real_ids.add((id2, id1))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    return list(candidate_pairs_real_ids)


def block_with_attr(X: DataFrame, attr):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    window_slide = 50
    X = X.sort_values(by=['NER_TEXT'])
    X.to_csv('t.csv')
    print(f'NER TIME : {time.time() - s}')
    candidate_pairs_real_ids = set()
    s = time.time()
    jeccard_threshold = 0.9
    window_size = 2
    for i in range(0, X.shape[0]-1):
        relevant_rows = X.iloc[i:i+window_size]
        pair = relevant_rows['NER'].values
        jecc = jeccard(set(pair[0]), set(pair[1]))
        if jecc > jeccard_threshold:
            ids = relevant_rows['id'].values
            id1 = ids[0]
            id2 = ids[1]
            if id1 > id2:
                candidate_pairs_real_ids.add((id2, id1))
            else:
                candidate_pairs_real_ids.add((id1, id2))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    return list(candidate_pairs_real_ids)


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


if __name__ == "__main__":
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv")
    # # perform blocking
    X1_candidate_pairs = block_with_attr_concept_2(X1, attr="title")
    X2_candidate_pairs = block_with_attr(X2, attr="name")
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    e = time.time()
    # print(f'CANDIDATE PAIRS X1 :{len(X1_candidate_pairs)}')
    # print(f'CANDIDATE PAIRS X2 :{len(X2_candidate_pairs)}')
    print(f'TOTAL TIME : {e - s}')
    #
    # print(f'X1 Recall: {compute_recall(X1_candidate_pairs, pd.read_csv("Y1.csv"))}')
    # print(f'X2 Recall: {compute_recall(X2_candidate_pairs, pd.read_csv("Y2.csv"))}')
