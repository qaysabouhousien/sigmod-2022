import itertools
import time
from collections import defaultdict
from difflib import get_close_matches
import pandas as pd
import spacy
from pandarallel import pandarallel
from pandas import DataFrame
import re

pandarallel.initialize(progress_bar=False, use_memory_fs=False)

ner = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'tok2vec', 'lemmatizer', 'attribute_ruler'])


def apply_ner(text, min_length, max_length=50, token_separators_pattern=None):
    if len(text) < min_length:
        return []
    if token_separators_pattern:
        text = token_separators_pattern.sub(' ', text)
    entities = ner(text)
    entities_text = list()
    for entity in entities:
        if entity.is_stop:
            continue
        lower = entity.text.lower().strip()
        if min_length < len(lower) < max_length:
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


def clean_brand(brand):
    if type(brand) == float:
        return ''
    return ' '.join(brand.split()[:1]).lower()


def save_tokens(X):
    t = defaultdict(int)
    for i in range(X.shape[0]):
        tokens = X['NER'][i]
        for token in tokens:
            t[token] += 1
    df = DataFrame(columns=['token', 'cnt', 'len'])

    for i, k in enumerate(t):
        df.loc[i] = [k, t[k], len(k)]
    df.to_csv('x2_tokens.csv', index=False)


def block_X2(X: DataFrame, jeccard_lower_threshold, jeccard_upper_threshold, jeccard_tolerance, max_block_size,
             min_token_length, max_token_length):
    s = time.time()
    pattern = re.compile(r"&NBSP;|&nbsp;|\\n|&amp|[=+><()\[\]{}/\\_&#?;,]|\.{2,}")
    X['NER'] = X['name'].parallel_apply(
        lambda x: apply_ner(x, min_token_length, max_token_length, pattern))
    save_tokens(X)
    print(f'NER TIME : {time.time() - s}')
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    X['NER_TEXT_ASC'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x, reverse=True)))
    X['NER_TEXT_MID'] = X['NER'].parallel_apply(lambda x: ' '.join(sort_from_mid(x)))
    sort_by_columns = ['NER_TEXT', 'NER_TEXT_ASC', 'NER_TEXT_REV', 'NER_TEXT_MID', 'price', 'brand']
    return block_dataset(X, sort_by_columns, max_block_size, jeccard_tolerance, jeccard_lower_threshold,
                            jeccard_upper_threshold)


def block_X1(X: DataFrame, jeccard_lower_threshold, jeccard_upper_threshold, jeccard_tolerance, max_block_size,
             min_token_length, max_token_length):
    s = time.time()
    pattern = None  # r"[\(\);,\\\/#\"':+]"
    X['NER'] = X['title'].parallel_apply(lambda x: apply_ner(x, min_token_length, max_token_length, pattern))
    print(f'NER TIME : {time.time() - s}')
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    X['NER_TEXT_ASC'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x, reverse=True)))

    sort_by_columns = ['NER_TEXT', 'NER_TEXT_ASC', 'NER_TEXT_REV', 'title']
    return block_dataset(X, sort_by_columns, max_block_size, jeccard_tolerance, jeccard_lower_threshold,
                         jeccard_upper_threshold)


def block_dataset(X, sort_by_columns, max_block_size, jeccard_tolerance, jeccard_lower_threshold,
                  jeccard_upper_threshold):
    s = time.time()
    candidate_pairs_real_ids = set()
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [(set(n[0]), n[1]) for n in X[['NER', 'id']].values]
        for i in range(len(block)):
            jeccard_passed = 0
            id1 = block[i][1]
            current_block_ner = block[i][0]
            block_size = 0
            for j in range(i + 1, len(block)):
                if block_size > max_block_size or jeccard_passed > jeccard_tolerance:
                    break
                similarity = jeccard(current_block_ner, block[j][0])
                if similarity < jeccard_lower_threshold:
                    jeccard_passed += 1
                    continue
                if similarity >= jeccard_upper_threshold:
                    id2 = block[j][1]
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


def eval_dataset(candidate_pairs, Y, dataset_name):
    print(f'CANDIDATE PAIRS {dataset_name} :{len(candidate_pairs)}')
    total_recall = round(compute_recall(candidate_pairs, Y), 3)
    within_size_recall = round(compute_recall(candidate_pairs[:len(Y)], Y), 3)
    total_precision = round(compute_precision(candidate_pairs, Y), 3)
    print(
        f'{dataset_name} Recall: {total_recall} RECALL WITHIN SIZE: {within_size_recall} PRECISION: {total_precision}')
    return [total_recall, within_size_recall, total_precision]


def evaluate(X1_candidate_pairs, X2_candidate_pairs):
    Y1 = pd.read_csv("Y1.csv")
    Y2 = pd.read_csv("Y2.csv")
    return [eval_dataset(X1_candidate_pairs, Y1, 'X1'), eval_dataset(X2_candidate_pairs, Y2, 'X2')]


def grid_search_X2():
    # .30, .40, 100, 150, 1, 10)
    jeccard_low = [(i / 100) + 0.28 for i in range(5)]
    jeccard_high = [(i / 100) + 0.38 for i in range(5)]
    jeccard_tolerance = [i * 10 + 80 for i in range(5)]
    max_block_size = [i * 10 + 130 for i in range(5)]
    min_token_length = [i for i in range(3)]
    max_token_length = [i + 9 for i in range(3)]
    grid = list(itertools.product(
        *[jeccard_low, jeccard_high, jeccard_tolerance, max_block_size, min_token_length, max_token_length]))
    print(len(grid))
    best_within_size = 0
    best_comp = []
    for i, comp in enumerate(grid):
        X2 = pd.read_csv("X2.csv", usecols=["id", "name", "price", "brand"])
        Y2 = pd.read_csv("Y2.csv")
        X2_candidate_pairs = block_X2(X2, comp[0], comp[1], comp[2], comp[3], comp[4], comp[5])
        eval_res = eval_dataset(X2_candidate_pairs, Y2, 'X2')
        within_size_recall = eval_res[1]
        if within_size_recall > best_within_size:
            best_within_size = within_size_recall
            best_comp = comp
        print(f'#{i} CURRENT WITHIN SIZE: {within_size_recall} params: {comp}')
        print(f'BEST WITHIN SIZE: {best_within_size} comp: {best_comp}')


def grid_search_X1():
    # .25, .40, 75, 15, 3, 100
    jeccard_low = [(i / 100) + 0.23 for i in range(5)]
    jeccard_high = [(i / 100) + 0.38 for i in range(5)]
    jeccard_tolerance = [i * 10 + 55 for i in range(5)]
    max_block_size = [i + 13 for i in range(5)]
    min_token_length = [i + 2 for i in range(3)]
    max_token_length = [i * 10 + 90 for i in range(3)]
    grid = list(itertools.product(
        *[jeccard_low, jeccard_high, jeccard_tolerance, max_block_size, min_token_length, max_token_length]))
    print(len(grid))
    best_within_size = 0
    best_comp = []
    for i, comp in enumerate(grid):
        X1 = pd.read_csv("X1.csv")
        Y1 = pd.read_csv("Y1.csv")
        candidate_pairs = block_X1(X1, comp[0], comp[1], comp[2], comp[3], comp[4], comp[5])
        eval_res = eval_dataset(candidate_pairs, Y1, 'X1')
        within_size_recall = eval_res[1]
        if within_size_recall > best_within_size:
            best_within_size = within_size_recall
            best_comp = comp
        print(f'#{i} CURRENT WITHIN SIZE: {within_size_recall} params: {comp}')
        print(f'BEST WITHIN SIZE: {best_within_size} comp: {best_comp}')


def run():
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv", usecols=["id", "name", "price", "brand"])
    # # perform blocking
    #  ORACLES :::
    # X1_candidate_pairs = block_with_attr(X1, "title", 0.06, 192)
    # X2_candidate_pairs = block_with_attr(X2, "name", 0, 1)
    X1_candidate_pairs = block_X1(X1, .23, .38, 55, 17, 3, 90)
    X2_candidate_pairs = block_X2(X2, .36, .36, 120, 130, 1, 10)
    # X2_candidate_pairs = block_X2(X2, .30, .40, 100, 150, 1, 10)

    # save results
    # save_output(X1_candidate_pairs, X2_candidate_pairs)
    evaluate(X1_candidate_pairs, X2_candidate_pairs)
    e = time.time()
    print(f'TOTAL TIME : {e - s}')


if __name__ == "__main__":
    # grid_search_X1()
    run()
