import itertools
import time
from collections import defaultdict
from difflib import get_close_matches

import pandas as pd
import spacy
import unidecode
from pandarallel import pandarallel
from pandas import DataFrame
import re

pandarallel.initialize(progress_bar=False, use_memory_fs=False)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tok2vec', 'ner'])

digs_pattern = re.compile(r'\d+')


def tokenize(text, min_length=1, max_length=50, token_separators_pattern=None):
    if type(text) != str:
        return []
    if len(text) < min_length:
        return []
    text = unidecode.unidecode(text)
    if token_separators_pattern:
        text = token_separators_pattern.sub(' ', text)
    entities = nlp(text)
    entities_text = list()
    pos = ['NOUN', 'PROPN']
    for entity in entities:
        if entity.pos_ not in pos or entity.is_stop:
            continue
        lower = entity.lemma_.lower().strip()
        if min_length < len(lower) < max_length:
            entities_text.append(lower)
    return entities_text


def jeccard(s1: set, s2: set):
    intersection = intersection_cardinality(s1, s2)
    union = len(s1) + len(s2) - intersection
    if union == 0:
        return 0
    return intersection / union


def intersection_cardinality(s1: set, s2: set):
    cardinality = 0
    s1_len = len(s1)
    s2_len = len(s2)
    small, large = (s2, s1) if s1_len > s2_len else (s1, s2)
    for v in small:
        if v in large:
            cardinality += 1
    return cardinality


def dice_coefficient(s1: set, s2: set):
    return 2 * intersection_cardinality(s1, s2) / (len(s1) + len(s2))


def overlap_coefficient(s1: set, s2: set):
    intersection = intersection_cardinality(s1, s2)
    d = min(len(s1), len(s2))
    if d == 0:
        return 0
    return intersection / d


def sort_from_index(x: [], index):
    if index >= len(x):
        return []
    return get_close_matches(x[index], x, cutoff=0, n=len(x))


def sort_from_mid(x: []):
    if len(x) < 3:
        return x
    return sort_from_index(x, len(x) // 2)


def clean_category(category):
    if type(category) == float:
        return ''
    return category


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


gb_pattern = re.compile(r'([\d]+)\s?g[ob]', re.IGNORECASE)
# gb_pattern = re.compile(r'([\d]+)\s?g[ob]', re.IGNORECASE)
pattern_2 = re.compile(r'[a-zA-Z]\d+')


def find_pattern(text, pattern):
    matches = pattern.findall(text)
    distinct = set()
    for m in matches:
        distinct.add(m.lower())
    return distinct


def block_X2(X: DataFrame, jeccard_threshold, jeccard_tolerance, min_token_length, max_token_length, gb_prop, p2_prop):
    s = time.time()
    pattern = re.compile(r"&NBSP;|&nbsp;|\\n|&amp|[=+><()\[\]{}/\\_&#?;,]|\.{2,}", re.IGNORECASE)
    X['tokens'] = X['name'].parallel_apply(lambda x: tokenize(x, min_token_length, max_token_length, pattern))
    print(f'TOKENIZATION TIME : {time.time() - s}')
    s = time.time()
    X['gb_p'] = X['name'].parallel_apply(lambda x: find_pattern(x, gb_pattern))
    X['p2'] = X['name'].parallel_apply(lambda x: find_pattern(x, pattern_2))
    print(f'FINDING PATTERNS TIME: {time.time() - s}')
    X['GBs_text'] = X['gb_p'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['tokens_text_asc'] = X['tokens'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['tokens_text_desc'] = X['tokens'].parallel_apply(lambda x: ' '.join(sorted(x, reverse=True)))
    sort_by_columns = ['tokens_text_asc', 'tokens_text_desc', 'GBs_text']
    return block_dataset_2(X, sort_by_columns, jeccard_tolerance, jeccard_threshold, gb_prop, p2_prop)


def block_X1(X: DataFrame, jeccard_hreshold, max_block_size, min_token_length, max_token_length, p2_prop):
    s = time.time()
    pattern = None  # re.compile(r'amazon\.com', re.IGNORECASE)
    X['tokens'] = X['title'].parallel_apply(lambda x: tokenize(x, min_token_length, max_token_length, pattern))
    print(f'TOKENIZATION TIME : {time.time() - s}')
    X['p2'] = X['title'].parallel_apply(lambda x: find_pattern(x, pattern_2))
    X['p2_text'] = X['p2'].parallel_apply(lambda x: ' '.join(sorted(x)))
    X['tokens_text_desc'] = X['tokens'].parallel_apply(lambda x: ' '.join(sorted(x, reverse=True)))
    sort_by_columns = ['tokens_text_desc', 'title', 'p2_text']
    return block_dataset(X, sort_by_columns, max_block_size, jeccard_hreshold, p2_prop)


def block_dataset_2(X, sort_by_columns, jeccard_tolerance, jeccard_threshold, gb_prop, p2_prop):
    """
    :param p2_prop:
    :param gb_prop:
    :param X: dataframe to create blocks on
    :param sort_by_columns: columns to sort the dataframe on
    :param jeccard_tolerance: initial tolerance limit, how many non matching records(i.e. lower that jeccard_threshold)
     are allowed before moving to the next iteration
    :param jeccard_threshold: minimum threshold for jeccard similarity to consider a pair as candidate match
    :return: list of record pairs that are considered as candidate matches
    """
    s = time.time()
    candidate_pairs_real_ids = set()
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1]), n[2], n[3], n[4]) for n in X[['id', 'tokens', 'gb_p', 'GBs_text', 'p2']].values]
        for i in range(len(block)):
            if column == 'GBs_text':
                sorted_i = block[i][3]
                if not sorted_i:
                    continue
            jeccard_passed = 0
            id1 = block[i][0]
            tokens_i = block[i][1]
            gbs_i = block[i][2]
            p2_i = block[i][4]
            has_gbs = len(gbs_i) > 0
            has_p2 = len(p2_i) > 0
            block_size = 0
            for j in range(i + 1, len(block)):
                normal_sim_prop = 1
                if jeccard_passed > jeccard_tolerance:
                    break
                tokens_j = block[j][1]
                filter_tokens = ['kingston', 'sandisk', 'samsung', 'toshiba', 'lexar', 'galaxy', 'sony']
                halt_iter = False
                for ft in filter_tokens:
                    if should_filter_on_token(ft, tokens_i, tokens_j):
                        halt_iter = True
                        break
                if halt_iter:
                    jeccard_passed += 1
                    continue
                similarity = jeccard(tokens_i, tokens_j)
                gbs_j = block[j][2]
                p2_j = block[j][4]
                similarity_GBs = 0
                if has_gbs or len(gbs_j) > 0:
                    similarity_GBs = jeccard(gbs_i, gbs_j)
                    normal_sim_prop -= gb_prop
                similarity_p2 = 0
                if has_p2 or len(p2_j) > 0:
                    similarity_p2 = jeccard(p2_i, p2_j)
                    normal_sim_prop -= p2_prop
                similarity = similarity_GBs * gb_prop + similarity * normal_sim_prop + similarity_p2 * p2_prop
                if similarity < jeccard_threshold:
                    jeccard_passed += 1
                    continue
                id2 = block[j][0]
                # order the pair by having the smaller id first.
                pair = (id1, id2) if id2 > id1 else (id2, id1)
                candidate_pairs_real_ids.add((pair, similarity))
                block_size += 1
            # auto adapting the jeccard tolerance size dependent on how many pairs were found in the current block.
            # if the found_ratio is higher that threshold, then we increase the tolerance, otherwise, we decrease it.
            found_ratio = block_size / jeccard_tolerance
            threshold = 0.05
            if found_ratio > threshold:
                jeccard_tolerance = min(200, jeccard_tolerance + found_ratio * jeccard_tolerance)
            else:
                jeccard_tolerance = max(10, jeccard_tolerance - (threshold - found_ratio) * jeccard_tolerance)
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    # sorting by the similarity score to get the highest scores first.
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]


def should_filter_on_token(search, tokens_i, tokens_j):
    return (search in tokens_j and search not in tokens_i) or (search not in tokens_j and search in tokens_i)


def block_dataset(X, sort_by_columns, max_block_size, jeccard_threshold, p2_prop):
    """
    :param p2_prop:
    :param X: dataframe to create blocks on
    :param sort_by_columns: columns to sort the dataframe on
    :param max_block_size: initial max_block_size
    :param jeccard_threshold: minimum threshold for jeccard similarity to consider a pair as candidate match
    :return:
    """
    s = time.time()
    # set how much each set's similarity matters by setting a proportion to each one.
    candidate_pairs_real_ids = set()
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1]), n[2], n[3]) for n in X[['id', 'tokens', 'p2', 'p2_text']].values]
        for i in range(len(block)):
            if column == 'p2_text':
                sorted_i = block[i][3]
                if not sorted_i:
                    continue
            id1 = block[i][0]
            all_i = block[i][1]
            p2_i = block[i][2]
            has_p2 = len(p2_i) > 0
            block_size = 0
            for j in range(i + 1, min(len(block), i + max_block_size)):
                id2 = block[j][0]
                normal_sim_prop = 1
                if block_size > max_block_size:
                    break
                all_j = block[j][1]
                similarity = jeccard(all_i, all_j)
                p2_j = block[j][2]
                p2_sim = 0
                if has_p2 or len(p2_j) > 0:
                    p2_sim = jeccard(p2_i, p2_j)
                    normal_sim_prop -= p2_prop
                similarity = p2_sim * p2_prop + similarity * normal_sim_prop
                if similarity < jeccard_threshold:
                    continue
                pair = (id1, id2) if id2 > id1 else (id2, id1)
                candidate_pairs_real_ids.add((pair, similarity))
                block_size += 1
            # auto adapting the max block size dependent on how many pairs were found in the current block.
            # if the found_ratio is higher that threshold, then we enlarge the size, otherwise, we shrink it.
            found_ratio = block_size / max_block_size
            thres = 0.05
            if found_ratio > thres:
                max_block_size = min(200, max_block_size + int(found_ratio * max_block_size))
            else:
                max_block_size = max(10, max_block_size - int((thres - found_ratio) * max_block_size))

    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    # sorting by the similarity score to get the highest scores first.
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
    total_recall = compute_recall(candidate_pairs, Y)
    within_size_recall = compute_recall(candidate_pairs[:len(Y)], Y)
    total_precision = compute_precision(candidate_pairs, Y)
    print(f'{dataset_name} Recall: {round(total_recall, 3)} '
          f'RECALL WITHIN SIZE: {round(within_size_recall, 3)} PRECISION: {round(total_precision, 3)}')
    return [total_recall, within_size_recall, total_precision]


def evaluate(X1_candidate_pairs, X2_candidate_pairs):
    Y1 = pd.read_csv("Y1.csv")
    Y2 = pd.read_csv("Y2.csv")
    return [eval_dataset(X1_candidate_pairs, Y1, 'X1'), eval_dataset(X2_candidate_pairs, Y2, 'X2')]


def prop_grid_search_x1():
    p2 = [i / 100 for i in range(40)]
    best_within_size = 0
    best_comp = []
    for i, comp in enumerate(p2):
        X1 = pd.read_csv("X1.csv")
        Y1 = pd.read_csv("Y1.csv")
        candidate_pairs = block_X1(X1, .38, 200, 3, 90, comp)
        eval_res = eval_dataset(candidate_pairs, Y1, 'X1')
        within_size_recall = eval_res[0]
        if within_size_recall > best_within_size:
            best_within_size = within_size_recall
            best_comp = comp
        print(f'#{i} CURRENT WITHIN SIZE: {within_size_recall} params: {comp}')
        print(f'BEST WITHIN SIZE: {best_within_size} comp: {best_comp}')


def prop_grid_search_x2():
    gb = [i / 100 for i in range(30)]
    p2 = [i / 100 for i in range(30)]
    grid = list(itertools.product(*[gb, p2]))
    best_within_size = 0
    best_comp = []
    for i, comp in enumerate(grid):
        X2 = pd.read_csv("X2.csv", usecols=["id", "name", "price", "brand"])
        Y2 = pd.read_csv("Y2.csv")
        X2_candidate_pairs = block_X2(X2, .40, 200, 1, 10, comp[0], comp[1])
        eval_res = eval_dataset(X2_candidate_pairs, Y2, 'X2')
        within_size_recall = eval_res[1]
        if within_size_recall > best_within_size:
            best_within_size = within_size_recall
            best_comp = comp
        print(f'#{i} CURRENT WITHIN SIZE: {within_size_recall} params: {comp}')
        print(f'BEST WITHIN SIZE: {best_within_size} comp: {best_comp}')


def grid_search_X2():
    # .30, .40, 100, 150, 1, 10)
    # X2, .40, 200, 1, 10, .25, .08)
    jeccard_low = [(i / 100) + 0.28 for i in range(5)]
    jeccard_tolerance = [i * 10 + 80 for i in range(5)]
    max_block_size = [i * 10 + 130 for i in range(5)]
    min_token_length = [i for i in range(3)]
    max_token_length = [i + 9 for i in range(3)]
    grid = list(itertools.product(
        *[jeccard_low, jeccard_tolerance, max_block_size, min_token_length, max_token_length]))
    print(len(grid))
    best_within_size = 0
    best_comp = []
    for i, comp in enumerate(grid):
        X2 = pd.read_csv("X2.csv", usecols=["id", "name", "price", "brand"])
        Y2 = pd.read_csv("Y2.csv")
        X2_candidate_pairs = block_X2(X2, comp[0], comp[1], comp[2], comp[3], comp[4])
        eval_res = eval_dataset(X2_candidate_pairs, Y2, 'X2')
        within_size_recall = eval_res[1]
        if within_size_recall > best_within_size:
            best_within_size = within_size_recall
            best_comp = comp
        print(f'#{i} CURRENT WITHIN SIZE: {within_size_recall} params: {comp}')
        print(f'BEST WITHIN SIZE: {best_within_size} comp: {best_comp}')


def grid_search_X1():
    # X1, .38, 200, 3, 90)
    # jeccard_hreshold, max_block_size, min_token_length, max_token_length
    jeccard_low = [(i / 100) + 0.1 for i in range(50)]
    max_block_size = [i + 200 for i in range(1)]
    min_token_length = [i + 3 for i in range(1)]
    max_token_length = [i + 90 for i in range(1)]
    grid = list(itertools.product(
        *[jeccard_low, max_block_size, min_token_length, max_token_length]))
    print(len(grid))
    best_within_size = 0
    best_comp = []
    for i, comp in enumerate(grid):
        X1 = pd.read_csv("X1.csv")
        Y1 = pd.read_csv("Y1.csv")
        candidate_pairs = block_X1(X1, comp[0], comp[1], comp[2], comp[3])
        eval_res = eval_dataset(candidate_pairs, Y1, 'X1')
        within_size_recall = eval_res[1]
        if within_size_recall > best_within_size:
            best_within_size = within_size_recall
            best_comp = comp
        print(f'#{i} CURRENT WITHIN SIZE: {within_size_recall} params: {comp}')
        print(f'BEST WITHIN SIZE: {best_within_size} comp: {best_comp}')


def to_dict(candidate_pairs):
    d = defaultdict(set)
    for pair, sim in candidate_pairs:
        if sim > .96:
            d[pair[0]].add(pair[1])
            d[pair[1]].add(pair[0])
    return d


def transitive(candidate_pairs):
    d = to_dict(candidate_pairs)
    new_pairs = set()
    for pair in candidate_pairs:
        left = pair[0][0]
        right = pair[0][1]
        sim = pair[1]
        add_to_right = d[left]
        if len(add_to_right) > 1:
            for v in add_to_right:
                if v == right:
                    continue
                new_pair = (v, right) if right > v else (right, v)
                new_pairs.add((new_pair, sim))
        add_to_left = d[right]
        if len(add_to_left) > 1:
            for v in add_to_left:
                if v == left:
                    continue
                new_pair = (v, left) if left > v else (left, v)
                new_pairs.add((new_pair, sim))
    candidate_pairs |= new_pairs


def run():
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv", usecols=["id", "name"])
    # # perform blocking
    #  ORACLES :::
    # X1_candidate_pairs = block_X1(X1, .111, 90, 172, 3, 90)
    # X2_candidate_pairs = block_X2(X2, .0, 0, 465, 1, 10)
    X1_candidate_pairs = block_X1(X1, .35, 200, 3, 90, .35)
    X2_candidate_pairs = block_X2(X2, .45, 200, 1, 10, .30, .08)
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    # evaluate(X1_candidate_pairs, X2_candidate_pairs)
    e = time.time()
    print(f'TOTAL TIME : {e - s}')


if __name__ == "__main__":
    run()
