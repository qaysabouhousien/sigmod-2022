import xml.etree.ElementTree as ET
import pandas as pd


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
    return df


def read_y():
    y = pd.read_csv('data/cora_gold.csv', sep=';')
    return y


if __name__ == '__main__':
    y = read_y()
    print(y)
    # df = to_df()
    # df.to_csv('data/cora.csv')