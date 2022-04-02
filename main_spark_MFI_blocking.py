import re
import time

import pandas as pd
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.fpm import FPGrowth

import spacy
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, StringType, IntegerType

spark = SparkSession.builder.appName("Sigmod 2022") \
    .config("spark.ui.enabled", "false") \
    .config("spark.executor.processTreeMetrics.enabled", "false") \
    .getOrCreate()
ner = spacy.load("ner_model_0", disable=['tagger', 'parser'])
stopwords = ner.Defaults.stop_words


def apply_ner(text):
    if not text:
        return []
    entities = ner(text)
    entities_text = set()
    for entity in entities:
        lower = entity.text.lower()
        if lower not in stopwords and not re.match(r'[^A-Za-z]', lower):
            entities_text.add(lower)
    return list(entities_text)


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


def block_MFI(df: DataFrame) -> set:
    new_df = df.withColumn('ner', udf(apply_ner, ArrayType(StringType()))("title")).cache()
    dd = new_df.select("id", 'ner').rdd.collect()
    dd = [(d[0], set(d[1])) for d in dd]
    new_df.show()
    fps = time.time()

    def block_frequent_set(items: list):
        itemsSet = set(items)
        records = sorted([r[0] for r in dd if itemsSet.issubset(r[1])])
        l = list()
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                l.append((records[i], records[j]))
        return l

    lowest_support = 0.0001
    support = 0.2
    support_step = 0.01
    discovered_pairs = set()
    while support > lowest_support and new_df.count() > 0:
        fp_growth = FPGrowth(itemsCol="ner", minSupport=support)
        model = fp_growth.fit(new_df)
        print(f'DF Size: {new_df.count()}')
        print(f'Frequent Items size : {model.freqItemsets.count()}')
        # .filter("size(items) > 5")
        m = model.freqItemsets.withColumn("records",
                                          udf(block_frequent_set, ArrayType(ArrayType(IntegerType())))('items')).cache()
        m.show()
        res = m.select("records").rdd.map(lambda x: x[0]).collect()
        discovered_pairs |= {tuple(pair) for block_records in res for pair in block_records}
        left = [pair[0] for pair in discovered_pairs]
        right = [pair[1] for pair in discovered_pairs]
        new_df = new_df.filter(~df.id.isin(left) & ~df.id.isin(right))
        support -= support_step
    return discovered_pairs


def block(df: DataFrame, attr) -> set:
    ners = time.time()
    df = df.withColumn('ner', udf(apply_ner, ArrayType(StringType()))(attr)).cache()
    df.show()
    print(f'NER TIME: {time.time() - ners}')
    dd = df.select("id", 'ner').rdd.collect()
    dd = [(d[0], set(d[1])) for d in dd]
    fps = time.time()

    def block_frequent_set(items: list):
        itemsSet = set(items)
        records = sorted([r[0] for r in dd if itemsSet.issubset(r[1])])
        l = list()
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                l.append((records[i], records[j]))
        return l

    m = FPGrowth(itemsCol="ner", minSupport=0.01).fit(df).freqItemsets.filter("size(items) > 2").withColumn(
        "records",udf(block_frequent_set, ArrayType(ArrayType(IntegerType())))('items')).cache()
    res = m.select("records").rdd.map(lambda x: x[0]).collect()
    print(f'FPGrowth time {time.time() - fps}')
    return {tuple(pair) for block_records in res for pair in block_records}


if __name__ == "__main__":
    s = time.time()
    # X1 = spark.read.csv('X1.csv', header=True, inferSchema=True)
    # X1 = X1.withColumn("id", X1["id"].cast(IntegerType()))
    X2 = spark.read.csv('X2.csv', header=True, inferSchema=True)
    X2 = X2.withColumn("id", X2["id"].cast(IntegerType()))
    # X1_candidate_pairs = block(X1, 'title')
    X2_candidate_pairs = block(X2, 'name')
    # print(f"NUMBER OF CANDIDATE PAIRS X1: {len(X1_candidate_pairs)}")
    print(f"NUMBER OF CANDIDATE PAIRS X2: {len(X2_candidate_pairs)}")
    # e = time.time()
    # print(f'TOTAL TIME : {e - s}')
    # rc1 = compute_recall(X1_candidate_pairs, pd.read_csv("Y1.csv"))
    # print(f'X1 Recall: {rc1}')
    rc2 = compute_recall(X2_candidate_pairs, pd.read_csv("Y2.csv"))
    print(f'X2 Recall: {rc2}')
    # print(f'Average Recall: {rc1 * 1 / 3 + rc2 * 2 / 3}')
