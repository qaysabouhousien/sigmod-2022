import warnings

import pandas as pd
import numpy as np
import spacy
import random
import csv
import json
from spacy.training.example import Example

data = pd.read_csv("catalogue/laptop.csv", index_col=None)
cols = data.columns
num_ent = len(data.columns)
ent_list = list(np.arange(num_ent))
prod_name = []  # list of all product names
prod_ann = []  # list of all the annotations
for i in range(len(data)):  # loop for each laptop
    idx_list = random.sample(ent_list, num_ent)  # shuffling indices
    cont = []
    ann = []
    ann_idx = 0  # pointer for annotating
    for j in range(num_ent):  # creating the jumbled product name
        col_num = idx_list[j]  # column number according jumbled column index
        val = data.iloc[i, col_num]  # value of the entity
        cont.append(val)  # appending list of entities into a single list
        ann.append((ann_idx, len(val) + ann_idx, cols[col_num]))  # annotations and entity name
        ann_idx = ann_idx + len(val) + 1  # updating the annotation pointer

    prod_name.append(' '.join(cont))  # complete phrase for each laptop
    prod_ann.append(ann)

prod = []
for i in range(len(data)):
    prod.append([prod_name[i], prod_ann[i]])
prod_data = pd.DataFrame(prod, columns=['ProdName', 'Annotations'])


def convert_to_spacytrain(df):
    training_data = []
    for i in range(df.shape[0]):
        text = df['ProdName'][i]
        entities = df['Annotations'][i]
        training_data.append((text, {"entities": entities}))
    return training_data


train_data = convert_to_spacytrain(prod_data)


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        # ner = nlp.create_pipe()
        ner = nlp.add_pipe('ner', last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings(): # only train NER
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                # print(text, annotations)
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example],  # batch of annotations
                           drop=0.2,  # dropout - make it harder to memorise data
                           sgd=optimizer,  # callable to update weights
                           losses=losses)
            print(losses)
            if losses['ner'] < 50:
                break
    return nlp


prdnlp = train_spacy(train_data, 20)
prdnlp.to_disk('./ner_model_1')
