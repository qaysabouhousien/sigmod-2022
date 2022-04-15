import time
from difflib import get_close_matches

import pandas as pd
import spacy
from pandarallel import pandarallel
from pandas import DataFrame

# import re

pandarallel.initialize(progress_bar=True, use_memory_fs=False)

ner = spacy.load("ner_model_0", disable=['tagger', 'parser', 'ner', 'tok2vec', 'lemmatizer', 'attribute_ruler'])
stopwords = ner.Defaults.stop_words


def apply_ner(text, min_length):
    entities = ner(text)
    entities_text = list()
    for entity in entities:
        lower = entity.text.lower()
        if lower not in stopwords and len(lower) > min_length:
            entities_text.append(lower)
    return entities_text


def jeccard(s1: set, s2: set):
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 0
    return intersection / union


def sort_from_mid(x: []):
    if len(x) < 3:
        return x
    return get_close_matches(x[len(x) // 2], x, cutoff=0, n=len(x))


def block_X2(X: DataFrame, jeccard_threshold, jeccard_tolerance, max_block_size, min_token_length):
    s = time.time()
    X['NER'] = (X['name'] + " " + X['description'].fillna('').astype(str)).parallel_apply(
        lambda x: apply_ner(x, min_token_length))
    print(f'NER TIME : {time.time() - s}')
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    X['NER_TEXT_ASC'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    X['NER_TEXT_MID'] = X['NER'].parallel_apply(lambda x: ' '.join(sort_from_mid(x)))
    sort_by_columns = ['NER_TEXT', 'NER_TEXT_ASC', 'NER_TEXT_REV', 'NER_TEXT_MID', 'price', 'brand']
    return block_dataset(X, sort_by_columns, max_block_size, jeccard_tolerance, jeccard_threshold)


def block_X1(X: DataFrame, jeccard_threshold, jeccard_tolerance, max_block_size, min_token_length):
    s = time.time()
    X['NER'] = X['title'].parallel_apply(lambda x: apply_ner(x, min_token_length))
    print(f'NER TIME : {time.time() - s}')
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    X['NER_TEXT_ASC'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    sort_by_columns = ['NER_TEXT', 'NER_TEXT_ASC', 'NER_TEXT_REV']
    return block_dataset(X, sort_by_columns, max_block_size, jeccard_tolerance, jeccard_threshold)


def block_dataset(X, sort_by_columns, max_block_size, jeccard_tolerance, jeccard_threshold):
    s = time.time()
    candidate_pairs_real_ids = set()
    for column in sort_by_columns:
        X = X.sort_values(by=[column])
        block = X['id'].values
        block_ner = [set(n) for n in X['NER'].values]
        for i in range(len(block)):
            jeccard_passed = 0
            id1 = block[i]
            current_block_ner = block_ner[i]
            block_size = 0
            for j in range(i + 1, len(block)):
                if block_size > max_block_size:
                    break
                if jeccard_passed > jeccard_tolerance:
                    break
                similarity = jeccard(current_block_ner, block_ner[j])
                if similarity < jeccard_threshold:
                    jeccard_passed += 1
                    continue
                id2 = block[j]
                pair = (id1, id2) if id2 > id1 else (id2, id1)
                candidate_pairs_real_ids.add((pair, similarity))
                block_size += 1
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


def evaluate(X1_candidate_pairs, X2_candidate_pairs):
    print(f'CANDIDATE PAIRS X1 :{len(X1_candidate_pairs)}')
    print(f'CANDIDATE PAIRS X2 :{len(X2_candidate_pairs)}')
    Y1 = pd.read_csv("Y1.csv")
    Y2 = pd.read_csv("Y2.csv")
    print(
        f'X1 Recall: {round(compute_recall(X1_candidate_pairs, Y1), 3)} RECALL WITHIN SIZE: {round(compute_recall(X1_candidate_pairs[:len(Y1)], Y1), 3)} PRECISION: {round(compute_precision(X1_candidate_pairs, Y1), 3)}')
    print(
        f'X2 Recall: {round(compute_recall(X2_candidate_pairs, Y2), 3)} RECALL WITHIN SIZE: {round(compute_recall(X2_candidate_pairs[:len(Y2)], Y2), 3)} PRECISION: {round(compute_precision(X2_candidate_pairs, Y2), 3)}')


def run():
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv")

    # # perform blocking
    #  ORACLES :::
    # X1_candidate_pairs = block_with_attr(X1, "title", 0.06, 192)
    # X2_candidate_pairs = block_with_attr(X2, "name", 0, 1)
    X1_candidate_pairs = block_X1(X1, .25, 50, 15, 3)
    X2_candidate_pairs = block_X2(X2, .20, 25, 75, 1)
    # # save results
    # save_output(X1_candidate_pairs, X2_candidate_pairs)
    evaluate(X1_candidate_pairs, X2_candidate_pairs)
    e = time.time()
    print(f'TOTAL TIME : {e - s}')


if __name__ == "__main__":
    run()
