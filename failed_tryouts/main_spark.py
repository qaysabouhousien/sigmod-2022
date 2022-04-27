import re
import time
from itertools import permutations

import pandas as pd
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.fpm import FPGrowth

import spacy
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, IntegerType

spark = SparkSession.builder.appName("Sigmod 2022") \
    .config("spark.ui.enabled", "false") \
    .config("spark.executor.processTreeMetrics.enabled", "false") \
    .getOrCreate()
ner = spacy.load("../ner_model_0", disable=['tagger', 'parser'])
stopwords = ner.Defaults.stop_words


def apply_ner(text):
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


if __name__ == "__main__":

    s = time.time()
    X1 = spark.read.csv('X1.csv', header=True, inferSchema=True)
    nerS = time.time()
    new_df = X1.withColumn('ner', udf(apply_ner, ArrayType(StringType()))("title")).cache()
    print(f'NER TIME : {time.time() - nerS}')
    dd = new_df.select("id", 'ner').rdd.collect()
    new_df.show()
    # new_df = new_df.select('id', explode(new_df.ner)).groupBy('col').agg(collect_set("id"))
    # new_df = new_df.withColumnRenamed("collect_set(id)","ids")
    # new_df.show()
    fps = time.time()
    fpGrowth = FPGrowth(itemsCol="ner", minSupport=0.001)
    model = fpGrowth.fit(new_df)
    # Display frequent itemsets.
    model.freqItemsets.orderBy("freq", ascending=False).show(1000, False)
    print(f'FPGrowth time {time.time() - fps}')
    e = time.time()


    def block(items):
        return sorted([r[0] for r in dd if all([item in r[1] for item in items])])


    m = model.freqItemsets.filter("size(items) > 10").withColumn("records",
                                                                udf(block, ArrayType(IntegerType()))('items')).cache()
    res = m.select("records").rdd.map(lambda x: x[0]).collect()
    l = set()
    for block in res:
        records = sorted(block)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                l.add((records[i], records[j]))
    print(f"NUMBER OF CANDIDATE PAIRS: {len(l)}")
    print(f'TOTAL TIME : {e - s}')
    r = compute_recall(l, pd.read_csv("../Y1.csv"))
    print(r)
