from functools import lru_cache

import spacy
import unidecode
from pandas import DataFrame

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


def sn(X: DataFrame, sort_by_columns, window_size, YSet):
    candidate_pairs_real_ids = set()
    comparisons = 0
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [n for n in X['id'].values]
        for i in range(len(block)):
            id1 = block[i]
            block_size = 0
            for j in range(i + 1, min(len(block), i + window_size)):
                comparisons += 1
                id2 = block[j]
                if frozenset({id1, id2}) in YSet or frozenset({id2, id1}) in YSet:
                    pair = (id1, id2)
                    pair2 = (id2, id1)
                    candidate_pairs_real_ids.add(pair)
                    candidate_pairs_real_ids.add(pair2)
                    block_size += 1
    return list(candidate_pairs_real_ids), comparisons


def sn_jaccard(X: DataFrame, sort_by_columns, window_size, jeccard_threshold):
    candidate_pairs_real_ids = set()
    comparisons = 0
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1])) for n in X[['id', 'tokens']].values]
        for i in range(len(block)):
            id1 = block[i][0]
            tokens1 = block[i][1]
            block_size = 0
            for j in range(i + 1, min(len(block), i + window_size)):
                comparisons += 1
                id2 = block[j][0]
                tokens2 = block[j][1]
                if jeccard(tokens1, tokens2) > jeccard_threshold:
                    pair = (id1, id2)
                    pair2 = (id2, id1)
                    candidate_pairs_real_ids.add(pair)
                    candidate_pairs_real_ids.add(pair2)
                    block_size += 1
    return list(candidate_pairs_real_ids), comparisons


def dsc(X: DataFrame, sort_by_columns, initial_window_size, ratio_threshold, YSet):
    candidate_pairs_real_ids = set()
    comparisons = 0
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [n for n in X['id'].values]
        window = block[:initial_window_size]
        for j in range(len(X)):
            numDuplicates = 0
            numComparisons = 0
            k = 1
            while k < len(window):
                numComparisons += 1
                r1, r2 = window[0], window[k]
                if frozenset({r1, r2}) in YSet or frozenset({r2, r1}) in YSet:
                    numDuplicates += 1
                    pair = (r1, r2)
                    pair2 = (r2, r1)
                    candidate_pairs_real_ids.add(pair)
                    candidate_pairs_real_ids.add(pair2)
                if k == len(window) - 1 and j + k < len(X) and numDuplicates / numComparisons > ratio_threshold:
                    window.append(block[j + k])
                k += 1
            comparisons += numComparisons
            window = window[1:]
            if len(window) < initial_window_size and j + k < len(X):
                window.append(block[j + k])
            else:
                while len(window) > initial_window_size:
                    window = window[:-1]
    return list(candidate_pairs_real_ids), comparisons


def dsc_jeccard(X: DataFrame, sort_by_columns, initial_window_size, ratio_threshold, jeccard_threshold):
    candidate_pairs_real_ids = set()
    comparisons = 0
    for column in sort_by_columns:
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1])) for n in X[['id', 'tokens']].values]
        window = block[:initial_window_size]
        for j in range(len(X)):
            numDuplicates = 0
            numComparisons = 0
            k = 1
            while k < len(window):
                numComparisons += 1
                r1, r2 = window[0], window[k]
                id1 = r1[0]
                id2 = r2[0]
                if jeccard(r1[1], r2[1]) > jeccard_threshold:
                    numDuplicates += 1
                    pair = (id1, id2)
                    pair2 = (id1, id2)
                    candidate_pairs_real_ids.add(pair)
                    candidate_pairs_real_ids.add(pair2)
                if k == len(window) - 1 and j + k < len(X) and numDuplicates / numComparisons > ratio_threshold:
                    window.append(block[j + k])
                k += 1
            comparisons += numComparisons
            window = window[1:]
            if len(window) < initial_window_size and j + k < len(X):
                window.append(block[j + k])
            else:
                while len(window) > initial_window_size:
                    window = window[:-1]
    return list(candidate_pairs_real_ids), comparisons


def tolerance_jeccard(X, sort_by_columns, init_tol, jeccard_threshold):
    candidate_pairs_real_ids = set()
    comparisons = 0
    for column in sort_by_columns:
        jeccard_tolerance = init_tol
        X = X.sort_values(by=column)
        block = [(n[0], set(n[1])) for n in X[['id', 'tokens']].values]
        for i in range(len(block)):
            jeccard_passed = 0
            id1 = block[i][0]
            block_size = 0
            for j in range(i + 1, len(block)):
                comparisons += 1
                if jeccard_passed > jeccard_tolerance:
                    break
                id2 = block[j][0]
                if jeccard(block[j][1], block[i][1]) <= jeccard_threshold:
                    jeccard_passed += 1
                    continue
                pair = (id1, id2)
                pair2 = (id2, id1)
                candidate_pairs_real_ids.add(pair)
                candidate_pairs_real_ids.add(pair2)
                block_size += 1
            found_ratio = block_size / jeccard_tolerance
            thres = 0.05
            if found_ratio > thres:
                jeccard_tolerance = min(init_tol * 2, jeccard_tolerance + found_ratio * jeccard_tolerance)
            else:
                jeccard_tolerance = max(init_tol // 2, jeccard_tolerance - (thres - found_ratio) * jeccard_tolerance)
    return list(candidate_pairs_real_ids), comparisons


def tolerance(X, sort_by_columns, init_tol, YSet):
    candidate_pairs_real_ids = set()
    comparisons = 0
    for column in sort_by_columns:
        jeccard_tolerance = init_tol
        X = X.sort_values(by=column)
        block = [n for n in X['id'].values]
        for i in range(len(block)):
            jeccard_passed = 0
            id1 = block[i]
            block_size = 0
            for j in range(i + 1, len(block)):
                comparisons += 1
                if jeccard_passed > jeccard_tolerance:
                    break
                id2 = block[j]
                if not frozenset({id1, id2}) in YSet and not frozenset({id2, id1}) in YSet:
                    jeccard_passed += 1
                    continue
                pair = (id1, id2)
                pair2 = (id2, id1)
                candidate_pairs_real_ids.add(pair)
                candidate_pairs_real_ids.add(pair2)
                block_size += 1
            found_ratio = block_size / jeccard_tolerance
            thres = 0.05
            if found_ratio > thres:
                jeccard_tolerance = min(init_tol * 2, jeccard_tolerance + found_ratio * jeccard_tolerance)
            else:
                jeccard_tolerance = max(init_tol // 2, jeccard_tolerance - (thres - found_ratio) * jeccard_tolerance)
    return list(candidate_pairs_real_ids), comparisons


def adaptive(X, sort_by_columns, YSet):
    candidate_pairs_real_ids = set()
    init_max_block_size = 27288
    comparisons = 0
    for column in sort_by_columns:
        max_block_size = init_max_block_size
        X = X.sort_values(by=column)
        block = [n for n in X['id'].values]
        for i in range(len(block)):
            id1 = block[i]
            block_size = 0
            for j in range(i + 1, min(len(block), i + max_block_size)):
                comparisons += 1
                id2 = block[j]
                if frozenset({id1, id2}) not in YSet:
                    continue
                pair = (id1, id2) if id2 > id1 else (id2, id1)
                candidate_pairs_real_ids.add((pair, 0))
                block_size += 1
                if block_size > max_block_size:
                    break
            found_ratio = block_size / max_block_size
            thres = 0.1
            if found_ratio > thres:
                max_block_size = min(225, max_block_size + int(found_ratio * max_block_size))
            else:
                max_block_size = max(130, max_block_size - int((thres - found_ratio) * max_block_size))
    print(f'NUMBER OF Comparrisions: {comparisons}')
    candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
    return [p[0] for p in candidate_pairs_real_ids]

# def dscPP(X: DataFrame, key, initial_window_size, ratio_threshold, similarity_threshold):
#     skip_records = set()
#     X = X.sort_values(by=key)
#     window = X.iloc[:initial_window_size]
#     candidate_pairs_real_ids = set()
#     for j in range(len(X)):
#         if window.iloc[0]['id'] not in skip_records:
#             numDuplicates = 0
#             numComparisons = 0
#             k = 1
#             while k < len(window):
#                 r1, r2 = window.iloc[0], window.iloc[k]
#                 similarity = jeccard(set(r1['tokens']), set(r2['tokens']))
#                 if similarity > similarity_threshold:
#                     numDuplicates += 1
#                     pair = (r1['id'], r2['id']) if r2['id'] > r1['id'] else (r2['id'], r1['id'])
#                     candidate_pairs_real_ids.add((pair, similarity))
#                     skip_records.add(r2['id'])
#                     while len(window) < k + initial_window_size - 1 and j + len(window) < len(X):
#                         window = pd.concat([window, X.iloc[j + len(window): j + len(window) + 1]])
#                 if k == len(window) and j + k < len(X) and numDuplicates / numComparisons > ratio_threshold:
#                     window = pd.concat([window, X.iloc[j + k: j + k + 1]])
#                 k += 1
#             window = window.iloc[1:]
#             if len(window) < initial_window_size and j + k < len(X):
#                 window = pd.concat([window, X.iloc[j + k:j + k + 1]])
#             else:
#                 while len(window) > initial_window_size:
#                     window = window.iloc[:len(window) - 1]
#     candidate_pairs_real_ids = sorted(candidate_pairs_real_ids, key=lambda x: x[1], reverse=True)
#     return [p[0] for p in candidate_pairs_real_ids]
