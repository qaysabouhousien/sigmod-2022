"""
python -m spacy download en_core_web_sm

pip install spacy pandas numpy jmespath openpyxl
"""

import spacy
import pandas as pd
import jmespath
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_excel('/mnt/files/input.xlsx')

titles = jmespath.search('@.*', df[['title', 'id']].T.to_dict())


def transform(obj):
    t = obj['title']
    i = obj['id']

    return (i, ':'.join(t.split(':')[1:]))


titles = list(map(transform, titles))

nlp = spacy.load("en_core_web_sm")
dataset = []
for i, title in titles:
    tokens = nlp(title)

    # compute the mean of all token vectors
    vector_mean = np.mean(np.array([token.vector for token in tokens]), axis=0).astype('double')

    # from here you can feed the vectors to a k-means algorithm.
    # each vector is length of 95
    dataset.append({'id': i, 'vec': vector_mean})

kmeans = KMeans(n_clusters=18, random_state=0).fit(np.array(jmespath.search('[].vec', dataset)))

predictions = kmeans.predict(jmespath.search('[].vec', dataset))
identifiers = jmespath.search('[].id', dataset)

result = []

for i, pred in zip(identifiers, predictions):
    result.append({'id': i, 'group': pred})

prediction = pd.DataFrame(result)

fin_result = df.merge(prediction, how='outer', left_on=['id'], right_on=['id'])[['id', 'title', 'group']]
fin_result.to_csv('/mnt/files/predictions.csv', index=False)
