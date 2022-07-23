import time

from recordlinkage import datasets
import pandas as pd

from algorithms import sn, dsc, tolerance, sn_jaccard, dsc_jeccard, tolerance_jeccard, tokenize
from evaluate import eval_dataset


def run_perfect():
    X, yy = datasets.load_febrl2(True)
    X['id'] = X.index
    Y = pd.DataFrame(columns=['lid', 'rid'])
    for v in range(len(yy)):
        Y.loc[v] = [yy[v][0], yy[v][1]]
    ySet = {frozenset(v) for v in Y.values}
    sort_by_columns = ['given_name', 'surname', 'street_number', 'address_1', 'suburb']
    for w in range(2, 100):
        X_candidate_pairs, sn_comps = sn(X, sort_by_columns, w, ySet)
        sn_r, _, __ = eval_dataset(X_candidate_pairs, Y)
        th = 1 / (w - 1)
        X_candidate_pairs, dsc_comps = dsc(X, sort_by_columns, w, th, ySet)
        dsc_r, _, __ = eval_dataset(X_candidate_pairs, Y)
        X_candidate_pairs, tol_comps = tolerance(X, sort_by_columns, w, ySet)
        tol_r, _, __ = eval_dataset(X_candidate_pairs, Y)
        print(f' W={w}, SN=({sn_comps}:{sn_r:.2f}), DSC=({dsc_comps}:{dsc_r:.2f}), TOL=({tol_comps}:{tol_r:.2f})')


def run_imperfect():
    X, yy = datasets.load_febrl2(True)
    print(X.columns)
    X['id'] = X.index
    Y = pd.DataFrame(columns=['lid', 'rid'])
    for v in range(len(yy)):
        Y.loc[v] = [yy[v][0], yy[v][1]]
    s = time.time()
    X['tokens'] = X['address_1'].apply(tokenize)
    print(f'TOKENIZATION TIME : {time.time() - s}')
    X['TOKENS_TEXT_ASC'] = X['tokens'].apply(lambda x: ' '.join(sorted(x)))
    X['TOKENS_TEXT_DESC'] = X['tokens'].apply(lambda x: ' '.join(sorted(x, reverse=True)))
    sort_by_columns = ['TOKENS_TEXT_ASC', 'TOKENS_TEXT_DESC']
    w = 50
    print('threshold,SN_COMPS,SN_RECALL,SN_PRECISION,DSC_COMPS,DSC_RECALL,DSC_PRECISION,TOL_COMPS,TOL_RECALL,TOL_PRECISION')
    for i in range(3, 10):
        jaccard_threshold = 0.1 * i
        X_candidate_pairs, sn_comps = sn_jaccard(X, sort_by_columns, w, jaccard_threshold)
        sn_r, _, sn_p = eval_dataset(X_candidate_pairs, Y)
        th = 1 / (w - 1)
        X_candidate_pairs, dsc_comps = dsc_jeccard(X, sort_by_columns, w, th, jaccard_threshold)
        dsc_r, _, dsc_p = eval_dataset(X_candidate_pairs, Y)
        X_candidate_pairs, tol_comps = tolerance_jeccard(X, sort_by_columns, w, jaccard_threshold)
        tol_r, _, tol_p = eval_dataset(X_candidate_pairs, Y)
        print(
            f'{jaccard_threshold:.1f},{sn_comps},{sn_r:.2f},{sn_p:.2f},{dsc_comps},{dsc_r:.2f},{dsc_p:.2f},{tol_comps},{tol_r:.2f},{tol_p}')


if __name__ == "__main__":
    run_imperfect()