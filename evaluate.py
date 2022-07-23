

def compute_precision(X, Y):
    c = 0
    st = {tuple(i) for i in Y.to_numpy()}
    for pair in X:
        if pair in st:
            c += 1
    return c / len(X)


def compute_recall(X, Y):
    c = 0
    st = set(X)
    for i in range(Y.shape[0]):
        t = (Y['lid'][i], Y['rid'][i])
        if t in st:
            c += 1
    return c / len(Y)


def eval_dataset(candidate_pairs, Y):
    total_recall = compute_recall(candidate_pairs, Y)
    within_size_recall = compute_recall(candidate_pairs[:len(Y)], Y)
    total_precision = compute_precision(candidate_pairs, Y)
    return [total_recall, within_size_recall, total_precision]