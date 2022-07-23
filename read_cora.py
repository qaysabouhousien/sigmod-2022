import xml.etree.ElementTree as ET
import pandas as pd
from pandas import DataFrame


def to_df():
    root = ET.parse('data/CORA.xml').getroot()
    elements = {}
    for ref in root.findall('NEWREFERENCE'):
        ref_id = ref.get('id')
        tags = {e.tag for e in list(ref.iter())}
        tags_texts = {}
        for tag in tags:
            val = ref.find(tag)
            if val is not None:
                tags_texts[tag] = val.text
        elements[ref_id] = tags_texts
    df = pd.DataFrame.from_dict(elements, orient='index')
    df['id'] = pd.to_numeric(df.index)
    return df


def read_y() -> DataFrame:
    y = pd.read_csv('data/cora_gold.csv', sep=';').rename(columns ={'id1' : 'lid', 'id2' : 'rid'})
    return y


if __name__ == '__main__':
    df = to_df()
    print(df.columns)
