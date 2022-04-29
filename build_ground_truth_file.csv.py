import re

import pandas as pd

import spacy
import unidecode
from Levenshtein import jaro_winkler

ner = spacy.load("en_core_web_sm", disable=['parser', 'tok2vec', 'ner'])


def apply_ner(text, min_length=1, max_length=50, token_separators_pattern=None):
    if type(text) != str:
        return []
    if len(text) < min_length:
        return []
    text = unidecode.unidecode(text)
    if token_separators_pattern:
        text = token_separators_pattern.sub(' ', text)
    entities = ner(text)
    entities_text = list()
    pos = ['NOUN', 'PROPN']
    for entity in entities:
        if entity.pos_ not in pos or entity.is_stop:
            continue
        lower = entity.lemma_.lower().strip()
        if min_length < len(lower) < max_length:
            entities_text.append(lower)
    return entities_text


def save_true():
    X = pd.read_csv('X2.csv')
    Y = pd.read_csv('Y2.csv')
    with open('training_2.csv', 'w', encoding='utf-8') as f:
        line = ['name', 'price', 'brand', 'category', 'description', 'name_2', 'price_2', 'brand_2', 'category_2',
                'description_2']
        line = [f"\"{c}\"" for c in line]
        f.write(','.join(line) + "\n")
        for i in range(Y.shape[0]):
            l = Y['lid'][i]
            r = Y['rid'][i]
            ll = X[X['id'] == l].iloc[0]
            rr = X[X['id'] == r].iloc[0]
            line = [ll['name'], ll['price'], ll['brand'], ll['category'], ll['description']]
            line += [rr['name'], rr['price'], rr['brand'], rr['category'], rr['description']]
            line = [f"\"{c}\"" for c in line]
            f.write(','.join(line) + "\n")


def false_positive():
    X = pd.read_csv('X2.csv')
    Y = pd.read_csv('Y2.csv')
    out = pd.read_csv('output.csv', skiprows=range(1, 1_000_001))
    with open('training_3.csv', 'w', encoding='utf-8') as f:
        line = ['id', 'name', 'price', 'brand', 'category', 'description', 'id_2', 'name_2', 'price_2', 'brand_2',
                'category_2', 'description_2']
        line = [f"\"{c}\"" for c in line]
        f.write(','.join(line) + "\n")
        for i in range(out.shape[0]):
            l = out['left_instance_id'][i]
            if l == 0:
                break
            r = out['right_instance_id'][i]
            if len(Y[(Y['lid'] == l) & (Y['rid'] == r)]) == 0:
                ll = X[X['id'] == l].iloc[0]
                rr = X[X['id'] == r].iloc[0]
                line = [ll['id'], ll['name'], ll['price'], ll['brand'], ll['category'], ll['description']]
                line += [rr['id'], rr['name'], rr['price'], rr['brand'], rr['category'], rr['description']]
                line = [f"\"{c}\"" for c in line]
                f.write(','.join(line) + "\n")


def not_found():
    X = pd.read_csv('X2.csv')
    Y = pd.read_csv('Y2.csv')
    out = pd.read_csv('output.csv', skiprows=range(1, 1_000_001))
    with open('not_found.csv', 'w', encoding='utf-8') as f:
        line = ['id', 'name', 'price', 'brand', 'category', 'description', 'id_2', 'name_2', 'price_2', 'brand_2',
                'category_2', 'description_2']
        line = [f"\"{c}\"" for c in line]
        f.write(','.join(line) + "\n")
        for i in range(Y.shape[0]):
            l = Y['lid'][i]
            r = Y['rid'][i]
            if len(out[(out['left_instance_id'] == l) & (out['right_instance_id'] == r)]) == 0:
                ll = X[X['id'] == l].iloc[0]
                rr = X[X['id'] == r].iloc[0]
                line = [ll['id'], ll['name'], ll['price'], ll['brand'], ll['category'], ll['description']]
                line += [rr['id'], rr['name'], rr['price'], rr['brand'], rr['category'], rr['description']]
                line = [f"\"{c}\"" for c in line]
                f.write(','.join(line) + "\n")


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


def analyze_not_found():
    pattern_2 = re.compile(r'[a-zA-Z]+\d+')
    gb_pattern = re.compile(r'([\d]+)\s?(?:gb|GB)')
    v1 = 'Kingston DT101G2 Datatraveler Memoria USB portatile 32768 MB'
    v2 = 'Kingston DataTraveler DT101G2 32 GB USB-Stick USB 2.0 lila [Amazon Frustfreie Verpackung],Kingston,DT101G2/32GB'
    # df = pd.read_csv('not_found.csv')
    pattern = re.compile(r"&NBSP;|&nbsp;|\\n|&amp|[=+><()\[\]{}/\\_&#?;,]|\.{2,}")
    v1gb = set(gb_pattern.findall(v1))
    v2gb = set(gb_pattern.findall(v2))
    res = jeccard(set(apply_ner(v1, 1, 10, pattern)), set(apply_ner(v2, 1, 10, pattern)))
    res2 = jeccard(v1gb, v2gb)
    print(res, res2)
    # for v in res:
    #     print(v)


if __name__ == '__main__':
    analyze_not_found()
