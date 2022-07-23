from read_cora import read_y
import pandas as pd
from itertools import cycle

import matplotlib.pyplot as plt
from cycler import cycler

plt.rcParams.update({'font.size': 30})


def cora_clusters():
    y = read_y()
    r = y.groupby('lid').count()
    r = r.sort_values('rid')
    vals = r[r['rid'] > 1].values
    num_bins = 118
    plt.hist(vals, num_bins, rwidth=0.6)
    plt.ylabel('Cluster size')
    plt.xlabel('Cluster')
    plt.show()


def perfect_cora():
    sn = pd.read_csv('data/pref/perfect-cora/SN-pref.csv', header=None)
    dsc = pd.read_csv('data/pref/perfect-cora/DSC-pref.csv', header=None)
    tol = pd.read_csv('data/pref/perfect-cora/TOL-pref.csv', header=None)
    print(sn)
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(sn[2], sn[1], next(linecycler), label='SN', color='green')
    plt.plot(dsc[2], dsc[1], next(linecycler), label='DSC', color='red')
    plt.plot(tol[2], tol[1], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.ylabel('Number of comparisons')
    plt.xlabel('Recall')
    plt.yscale('log')
    plt.show()


def jeccard_cora_recall_threshold():
    df = pd.read_csv('data/pref/jeccard-cora/results.csv')
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(df['threshold'], df['SN_RECALL'], next(linecycler), label='SN', color='green')
    plt.plot(df['threshold'], df['DSC_RECALL'], next(linecycler), label='DSC', color='red')
    plt.plot(df['threshold'], df['TOL_RECALL'], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Recall')
    plt.show()


def jeccard_cora_precision_threshold():
    df = pd.read_csv('data/pref/jeccard-cora/results.csv')
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(df['threshold'], df['SN_PRECISION'], next(linecycler), label='SN', color='green')
    plt.plot(df['threshold'], df['DSC_PRECISION'], next(linecycler), label='DSC', color='red')
    plt.plot(df['threshold'], df['TOL_PRECISION'], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Precision')
    plt.show()


def jeccard_cora_f1_threshold():
    df = pd.read_csv('data/pref/jeccard-cora/results.csv')
    print(df.columns)
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(df['threshold'], df['SN_F1'], next(linecycler), label='SN', color='green')
    plt.plot(df['threshold'], df['DSC_F1'], next(linecycler), label='DSC', color='red')
    plt.plot(df['threshold'], df['TOL_F1'], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.xlabel('Similarity Threshold')
    plt.ylabel('F1-Score')
    plt.show()


def perfect_rest():
    sn = pd.read_csv('data/pref/perfect-rest/SN-pref.csv', header=None)
    dsc = pd.read_csv('data/pref/perfect-rest/DSC-pref.csv', header=None)
    tol = pd.read_csv('data/pref/perfect-rest/TOL-pref.csv', header=None)
    print(sn)
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(sn[2], sn[1], next(linecycler), label='SN', color='green')
    plt.plot(dsc[2], dsc[1], next(linecycler), label='DSC', color='red')
    plt.plot(tol[2], tol[1], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.ylabel('Number of comparisons')
    plt.xlabel('Recall')
    plt.show()


def perfect_febrl2():
    sn = pd.read_csv('data/pref/perfect-febrl2/SN.csv', header=None)
    dsc = pd.read_csv('data/pref/perfect-febrl2/DSC.csv', header=None)
    tol = pd.read_csv('data/pref/perfect-febrl2/TOL.csv', header=None)
    print(sn)
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(sn[2], sn[1], next(linecycler), label='SN', color='green')
    plt.plot(dsc[2], dsc[1], next(linecycler), label='DSC', color='red')
    plt.plot(tol[2], tol[1], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.ylabel('Number of comparisons')
    plt.xlabel('Recall')
    plt.ylim(100000, 250000)
    plt.xlim(0.96, 1)
    plt.show()



def jeccard_febrl_recall_threshold():
    df = pd.read_csv('data/pref/jeccard-febrl.csv')
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(df['threshold'], df['SN_RECALL'], next(linecycler), label='SN', color='green')
    plt.plot(df['threshold'], df['DSC_RECALL'], next(linecycler), label='DSC', color='red')
    plt.plot(df['threshold'], df['TOL_RECALL'], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Recall')
    plt.show()


def jeccard_febrl_precision_threshold():
    df = pd.read_csv('data/pref/jeccard-febrl.csv')
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(df['threshold'], df['SN_PRECISION'], next(linecycler), label='SN', color='green')
    plt.plot(df['threshold'], df['DSC_PRECISION'], next(linecycler), label='DSC', color='red')
    plt.plot(df['threshold'], df['TOL_PRECISION'], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Precision')
    plt.show()


def jeccard_febrl_f1_threshold():
    df = pd.read_csv('data/pref/jeccard-febrl.csv')

    print(df.columns)
    lines = ["k-", "k--", "k:"]
    linecycler = cycle(lines)
    plt.plot(df['threshold'], df['SN_F1'], next(linecycler), label='SN', color='green')
    plt.plot(df['threshold'], df['DSC_F1'], next(linecycler), label='DSC', color='red')
    plt.plot(df['threshold'], df['TOL_F1'], next(linecycler), label='Tolerance', color='blue')
    plt.legend()
    plt.xlabel('Similarity Threshold')
    plt.ylabel('F1-Score')
    plt.show()



if __name__ == "__main__":
    jeccard_febrl_recall_threshold()
