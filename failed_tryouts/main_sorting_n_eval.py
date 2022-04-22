import time
from difflib import get_close_matches

import pandas as pd
import spacy
from pandarallel import pandarallel
from pandas import DataFrame


pandarallel.initialize(progress_bar=True, use_memory_fs=False)

ner = spacy.load("../ner_model_0", disable=['tagger', 'parser', 'ner', 'tok2vec', 'lemmatizer', 'attribute_ruler'])
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


def block_with_attr_eval(X: DataFrame, attr, jeccard_threshold, jeccard_tolerance):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    X['NER_TEXT'] = X['NER'].parallel_apply(lambda x: ' '.join(x))
    X['NER_TEXT_ASC'] = X['NER'].parallel_apply(lambda x: ' '.join(sorted(x)))
    # sort NERs in DESC order
    X['NER_TEXT_REV'] = X['NER'].parallel_apply(lambda x: ' '.join(x[::-1]))
    X['NER_TEXT_MID'] = X['NER'].parallel_apply(lambda x: ' '.join(sort_from_mid(x)))
    candidate_pairs_real_ids = set()
    s = time.time()
    for op in range(4):
        if op == 0:
            X = X.sort_values(by=['NER_TEXT'])
        if op == 1:
            X = X.sort_values(by=['NER_TEXT_ASC'])
        if op == 2:
            X = X.sort_values(by=['NER_TEXT_REV'])
        if op == 3:
            X = X.sort_values(by=['NER_TEXT_MID'])
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
    return candidate_pairs_real_ids


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
    res = block_with_attr_eval(X1, 'title', 0.2, 20)
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
    r = [v[0] for v in res if v[1] == 1]
    print(len(r))
    print(f'TOTAL RECALL : {compute_recall(r, Y1)} TOTAL PRECISION: {compute_precision(r, Y1)}')
    c = 0
    st = {tuple(i) for i in Y1.to_numpy()}
    i = 0
    for pair in r[:100]:
        if pair not in st:
            print(X1[(X1['id'] == pair[0]) | (X1['id'] == pair[1])]['title'].values)
            print()


if __name__ == "__main__":
    eval()
