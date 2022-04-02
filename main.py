import re
import time

import spacy
import pandas as pd

# from pandarallel import pandarallel
import itertools

# pandarallel.initialize(progress_bar=True)

ner = spacy.load("ner_model_0", disable=['tagger', 'parser'])
stopwords = ner.Defaults.stop_words


def ner_test(X, attr):
    s = time.time()
    X['NER'] = X[attr].parallel_apply(ner)
    print(f'NER TIME : {time.time() - s}')
    records_entities = {}
    entities = {}
    for i in range(X.shape[0]):
        out = X['NER'][i]
        for entity in out.ents:
            text_lower = entity.text.lower()
            if text_lower in stopwords:
                continue
            if re.match(r'[^A-Za-z]', text_lower):
                continue
            if text_lower not in entities:
                entities[text_lower] = set()
            entities[text_lower].add(i)
            if i not in records_entities:
                records_entities[i] = set()
            records_entities[i].add(text_lower)
    print(records_entities)
    print(entities)


def jeccard_test(row, X, records_entities):
    ids = row.dropna()
    cp = list()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1 = ids[i]
            id2 = ids[j]
            # similarity = get_jaccard_cache(records_entities[id1], records_entities[id2])
            similarity = len(records_entities[id1].intersection(records_entities[id2])) / max(
                len(records_entities[id1]), len(records_entities[id2]))
            if similarity > 0.6:
                real_id1 = X['id'][id1]
                real_id2 = X['id'][id2]
                if real_id1 < real_id2:
                    t = (real_id1, real_id2)
                else:
                    t = (real_id2, real_id1)
                cp.append((t, similarity))
    if len(cp) == 0:
        return None
    return cp


def apply_ner(text):
    entities = ner(text)
    entities_text = []
    for entity in entities:
        lower = entity.text.lower()
        if lower not in stopwords and not re.match(r'[^A-Za-z]', lower):
            entities_text.append(lower)
    return entities_text


def block_with_attr_3(X, attr):
    entities = {}
    s = time.time()
    X['NER'] = X[attr].parallel_apply(apply_ner)
    print(f'NER TIME : {time.time() - s}')
    s = time.time()
    candidate_pairs_real_ids = set()
    for i in range(X.shape[0]):
        out = X['NER'][i]
        for entity in out:
            if entity not in entities:
                entities[entity] = set()
            entities[entity].add(i)
    entities = {k: v for k, v in entities.items() if 1 < len(v) < (len(X) * 0.05)}
    for _, records in entities.items():
        for i, j in itertools.combinations(records, 2):
            real_id1 = X['id'][i]
            real_id2 = X['id'][j]
            if real_id1 < real_id2:
                candidate_pairs_real_ids.add((real_id1, real_id2))
            else:
                candidate_pairs_real_ids.add((real_id2, real_id1))
    print(f'generating pairs {time.time() - s}')
    return list(candidate_pairs_real_ids)


def block_with_attr_2(X, attr, max_support):
    entities = {}
    s = time.time()
    X['NER'] = X[attr].apply(ner)
    print()
    print(f'NER TIME : {time.time() - s}')
    s = time.time()
    candidate_pairs_real_ids = set()
    for i in range(X.shape[0]):
        out = X['NER'][i]
        for entity in out.ents:
            text_lower = entity.text.lower()
            if text_lower in stopwords:
                continue
            if re.match(r'[^A-Za-z]', text_lower):
                continue
            if text_lower not in entities:
                entities[text_lower] = set()
            entities[text_lower].add(i)
    entities = {k: v for k, v in entities.items() if 1 < len(v) < (len(X) * max_support)}
    for _, records in entities.items():
        if len(records) < 1:
            continue
        for i, j in itertools.combinations(records, 2):
            real_id1 = X['id'][i]
            real_id2 = X['id'][j]
            if real_id1 < real_id2:
                candidate_pairs_real_ids.add((real_id1, real_id2))
            else:
                candidate_pairs_real_ids.add((real_id2, real_id1))
    print(f'generating pairs {time.time() - s} Size : {len(candidate_pairs_real_ids)}')
    return list(candidate_pairs_real_ids)
    # # for k, v in entities.items():
    # #     print(len(v), (len(X) * 0.05))
    # print(f'MAPPING RECORDS TO ENTITIES : {time.time() - s}')
    # s = time.time()
    # jaccard_similarities = []
    # df = pd.DataFrame.from_dict(entities, orient='index')
    # res = df.parallel_apply(lambda x: jeccard_test(x, X, records_entities), axis=1).to_frame()
    # print()
    # res = res[~res[0].isna()]
    # # l = list(itertools.chain.from_iterable(res.values))
    # l = [k for i in res.values for j in i for k in j]
    # from functools import cmp_to_key
    # def compare(x, y):
    #     return x[1] - y[1]
    #
    # l = sorted(l, key=cmp_to_key(compare), reverse=True)
    # print(f'JECCARD TIME: {time.time() - s}')
    # return [i[0] for i in l]
    #
    # candidate_pairs_real_ids = set()
    # for entity, records in entities.items():
    #     ids = list(sorted(records))
    #     if len(ids) < 1:
    #         continue
    #     for i in range(len(ids)):
    #         for j in range(i + 1, len(ids)):
    #             id1 = ids[i]
    #             id2 = ids[j]
    #             # similarity = get_jaccard_cache(records_entities[id1], records_entities[id2])
    #             similarity = len(records_entities[id1].intersection(records_entities[id2])) / max(
    #                 len(records_entities[id1]), len(records_entities[id2]))
    #             if similarity > 0.6:
    #                 real_id1 = X['id'][id1]
    #                 real_id2 = X['id'][id2]
    #                 if real_id1 < real_id2:
    #                     candidate_pairs_real_ids.add((real_id1, real_id2))
    #                 else:
    #                     candidate_pairs_real_ids.add((real_id2, real_id1))
    #                 jaccard_similarities.append(similarity)
    # print(f"size of pairs {len(candidate_pairs_real_ids)}")
    # candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    # print(f'JECCARD TIME: {time.time() - s}')
    #
    # return candidate_pairs_real_ids


def block_with_attr(X, attr):
    entities = {}
    s = time.time()
    X['NER'] = X[attr].parallel_apply(ner)
    print()
    print(f'NER TIME : {time.time() - s}')
    s = time.time()
    records_entities = {}
    for i in range(X.shape[0]):
        out = X['NER'][i]
        for entity in out.ents:
            text_lower = entity.text.lower()
            if text_lower in stopwords:
                continue
            if re.match(r'[^A-Za-z]', text_lower):
                continue
            if text_lower not in entities:
                entities[text_lower] = set()
            entities[text_lower].add(i)
            if i not in records_entities:
                records_entities[i] = set()
            records_entities[i].add(text_lower)
    entities = {k: v for k, v in entities.items() if 1 < len(v) < (len(X) * 0.05)}
    # for k, v in entities.items():
    #     print(len(v), (len(X) * 0.05))
    print(f'MAPPING RECORDS TO ENTITIES : {time.time() - s}')
    s = time.time()
    jaccard_similarities = []
    df = pd.DataFrame.from_dict(entities, orient='index')
    res = df.parallel_apply(lambda x: jeccard_test(x, X, records_entities), axis=1).to_frame()
    res = res[~res[0].isna()]
    # l = list(itertools.chain.from_iterable(res.values))
    l = [k for i in res.values for j in i for k in j]
    from functools import cmp_to_key
    def compare(x, y):
        return x[1] - y[1]

    l = sorted(l, key=cmp_to_key(compare), reverse=True)
    print(f'JECCARD TIME: {time.time() - s}')
    return [i[0] for i in l]
    #
    # candidate_pairs_real_ids = set()
    # for entity, records in entities.items():
    #     ids = list(sorted(records))
    #     if len(ids) < 1:
    #         continue
    #     for i in range(len(ids)):
    #         for j in range(i + 1, len(ids)):
    #             id1 = ids[i]
    #             id2 = ids[j]
    #             # similarity = get_jaccard_cache(records_entities[id1], records_entities[id2])
    #             similarity = len(records_entities[id1].intersection(records_entities[id2])) / max(
    #                 len(records_entities[id1]), len(records_entities[id2]))
    #             if similarity > 0.6:
    #                 real_id1 = X['id'][id1]
    #                 real_id2 = X['id'][id2]
    #                 if real_id1 < real_id2:
    #                     candidate_pairs_real_ids.add((real_id1, real_id2))
    #                 else:
    #                     candidate_pairs_real_ids.add((real_id2, real_id1))
    #                 jaccard_similarities.append(similarity)
    # print(f"size of pairs {len(candidate_pairs_real_ids)}")
    # candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    # print(f'JECCARD TIME: {time.time() - s}')
    #
    # return candidate_pairs_real_ids


jaccard_cache = {}


def get_jaccard_cache(s1, s2):
    t1 = frozenset(s1)
    t2 = frozenset(s2)
    if (t1, t2) in jaccard_cache:
        return jaccard_cache[(t1, t2)]
    j = len(t1.intersection(t2)) / max(len(t1), len(t2))
    jaccard_cache[(t1, t2)] = j
    return j


ner_cache = {}


def get_ner_cache(t):
    if t in ner_cache:
        return ner_cache[t]
    n = ner(t)
    ner_cache[t] = n
    return n


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
    return c / len(Y)


if __name__ == "__main__":
    s = time.time()
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv")
    # # perform blocking
    X1_candidate_pairs = block_with_attr_2(X1, attr="title", max_support=0.01)
    X2_candidate_pairs = block_with_attr_2(X2, attr="name", max_support=0.01)
    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    e = time.time()
    print(f'TOTAL TIME : {e - s}')

    print(f'X1 Recall: {compute_recall(X1_candidate_pairs, pd.read_csv("Y1.csv"))}')
    print(f'X2 Recall: {compute_recall(X2_candidate_pairs, pd.read_csv("Y2.csv"))}')
