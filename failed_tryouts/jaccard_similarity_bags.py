def multiset_intersection_cardinality(x: list, y: list) -> int:
    """Returns the number of elements of x and y intersection."""

    cardinality = 0
    fewest, most = (x, y) if len(x) < len(y) else (y, x)
    most = most.copy()
    for value in fewest:
        try:
            most.remove(value)
        except ValueError:
            pass
        else:
            cardinality += 1

    return cardinality


def multiset_union_cardinality(x: list, y: list) -> int:
    """Return the number of elements in both x and y."""
    return len(x) + len(y)


def jaccard_similarity_bags(x: list, y: list) -> float:
    """Get the Jaccard similarity of two bags (aka multisets).
    Example:
        >>> jaccard_similarity_bags([1,1,1,2], [1,1,2,2,3])
        0.3333333333333333
        >>> jaccard_similarity_bags([1,1,1,2], [1,2,3,4])
        0.25
        >>> jaccard_similarity_bags([1,1,2,2,3], [1,2,3,4])
        0.3333333333333333
    """

    intersection_cardinality = multiset_intersection_cardinality(x, y)
    union_cardinality = multiset_union_cardinality(x, y)
    return (
        intersection_cardinality
        / union_cardinality
    )