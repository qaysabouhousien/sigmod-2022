import pandas as pd

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

