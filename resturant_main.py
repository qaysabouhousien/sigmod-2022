import time
from difflib import get_close_matches
import pandas as pd
import spacy
import unidecode
from functools import lru_cache

from algorithms import sn, dsc, tolerance
from evaluate import eval_dataset

nlp = spacy.load("en_core_web_md", disable=['parser', 'tok2vec', 'ner'])


@lru_cache(maxsize=None)
def tokenize(text, min_length=1, token_separators_pattern=None):
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
        if min_length < len(lower):
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


def sort_from_index(x: [], index):
    if index >= len(x):
        return []
    return get_close_matches(x[index], x, cutoff=0, n=len(x))


def sort_from_mid(x: []):
    if len(x) < 3:
        return x
    return sort_from_index(x, len(x) // 2)


def block_dataset(X, sort_by_columns, jeccard_threshold):
    s = time.time()
    candidate_pairs_real_ids = set()
    max_block_size = 1000
    comparisons = 0
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1])) for n in X[['id', 'tokens']].values]
        for i in range(len(block)):
            jeccard_passed = 0
            id1 = block[i][0]
            all_i = block[i][1]
            block_size = 0
            for j in range(i + 1, min(len(block), i + max_block_size)):
                comparisons += 1
                id2 = block[j][0]
                if block_size > max_block_size:
                    break
                all_j = block[j][1]
                similarity = jeccard(all_i, all_j)
                if similarity < jeccard_threshold:
                    jeccard_passed += 1
                    continue
                pair = (id1, id2) if id2 > id1 else (id2, id1)
                candidate_pairs_real_ids.add((pair, similarity))
                block_size += 1
            found_ratio = block_size / max_block_size
            thres = 0.1
            if found_ratio > thres:
                max_block_size = min(200, max_block_size + int(found_ratio * max_block_size))
            else:
                max_block_size = max(10, max_block_size - int((thres - found_ratio) * max_block_size))
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    print(f'NUMBER OF Comparrisions: {comparisons}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]


def block_dataset_2(X, sort_by_columns, jeccard_threshold):
    s = time.time()
    candidate_pairs_real_ids = set()
    init_tol = 1000
    comparisons = 0
    for column in sort_by_columns:
        jeccard_tolerance = init_tol
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1])) for n in X[['id', 'tokens']].values]
        for i in range(len(block)):
            jeccard_passed = 0
            id1 = block[i][0]
            current_block_tokens = block[i][1]
            block_size = 0
            for j in range(i + 1, len(block)):
                if jeccard_passed > jeccard_tolerance:
                    break
                comparisons += 1
                similarity = jeccard(current_block_tokens, block[j][1])
                if similarity < jeccard_threshold:
                    jeccard_passed += 1
                    continue
                id2 = block[j][0]
                pair = (id1, id2) if id2 > id1 else (id2, id1)
                candidate_pairs_real_ids.add((pair, similarity))
                block_size += 1
            found_ratio = block_size / jeccard_tolerance
            thres = 0.05
            if found_ratio > thres:
                jeccard_tolerance = min(150, jeccard_tolerance + found_ratio * jeccard_tolerance)
            else:
                jeccard_tolerance = max(1, jeccard_tolerance - (thres - found_ratio) * jeccard_tolerance)
    print(f'SORTING NEIGHBORS TIME: {time.time() - s}')
    print(f'NUMBER OF Comparrisions: {comparisons}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]


def read_y(X):
    grouped = X.groupby('class')['id'].apply(list)
    grouped = grouped[grouped.apply(lambda x: len(x) > 1)].to_frame()
    grouped['lid'] = grouped['id'].apply(lambda x: x[0])
    grouped['rid'] = grouped['id'].apply(lambda x: x[1])
    grouped = grouped.reset_index().drop(columns=['class', 'id'])
    return grouped


def run():
    X = pd.read_csv("data/restaurant.original.csv")
    X['id'] = X.index
    Y = read_y(X)
    ySet = {frozenset(v) for v in Y.values}
    sort_by_columns = ['name', 'addr']
    for w in range(2, 250):
        X_candidate_pairs, sn_comps = sn(X, sort_by_columns, w, ySet)
        sn_r, _, __ = eval_dataset(X_candidate_pairs, Y)
        th = 1 / (w - 1)
        X_candidate_pairs, dsc_comps = dsc(X, sort_by_columns, w, th, ySet)
        dsc_r, _, __ = eval_dataset(X_candidate_pairs, Y)
        X_candidate_pairs, tol_comps = tolerance(X, sort_by_columns, w, ySet)
        tol_r, _, __ = eval_dataset(X_candidate_pairs, Y)
        print(f' W={w}, SN=({sn_comps}:{sn_r:.2f}), DSC=({dsc_comps}:{dsc_r:.2f}), TOL=({tol_comps}:{tol_r:.2f})')


if __name__ == "__main__":
    run()
