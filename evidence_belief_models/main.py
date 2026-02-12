from itertools import combinations
from .multi_layer_belief_model import MultiLayerBeliefModel
from .utils import get_key, get_set
from .topology import Topology, minimal_dense_set

def ds_int_belief_model(S, E):
    """
    Compute degrees of belief for non-empty subsets of S from evidence E.

    This reasoner applies the multi-layer belief model with:
    - Dempster–Shafer justification frame
    - intersection as allocation function

    It is equivalent to applying Dempster's rule of combination to E. 

    Only subsets with non-zero degree of belief are returned. The result
    is ordered first by increasing subset cardinality and, within each
    cardinality level, by decreasing degree of belief.

    Parameters
    ----------
    S : set 
        Set of possible states (frame of discernment).

    E : dict
        Quantitative evidence. Keys represent subsets of S (encoded using
        `get_key`), and values are belief masses associated with those
        subsets.

    Returns
    -------
    dict
        A dictionary mapping subsets of S (encoded using `get_key`) to
        their degree of belief. Only subsets with non-zero belief are
        included. The dictionary is ordered by increasing subset size and
        decreasing belief value.

    Example
    -------
    >>> S = {'dp', 'sp', 'dm'}
    >>> E = {get_key({'dp', 'dm'}): 0.8, get_key({'sp'}): 0.4}
    >>> result = ds_int_belief_model(S, E)
    >>> for k, b in result.items():
    ...     print(f"belief for {get_set(k)} -> {b:.2f}")
    >>> print("belief for other propositions -> 0")
    """
    model = MultiLayerBeliefModel(
        S,
        E,
        'dempster_shafer',
        set.intersection
    )

    beliefs = []

    # Generate all subsets of S
    for n in range(1, len(S) + 1):  # start from 1 to skip empty set
        for subset in combinations(S, n):
            p = set(subset)
            b = model.degree_of_belief(p)

            if b != 0:
                k = get_key(p)
                beliefs.append((k, b))

    # Sort:
    # 1) by subset size (ascending)
    # 2) by belief value (descending)
    beliefs.sort(key=lambda x: (len(get_set(x[0])), -x[1]))

    # Build ordered dictionary
    result = {k: b for k, b in beliefs}

    return result


def ds_min_belief_model(S, E):
    """
    Compute degrees of belief for non-empty subsets of S from evidence E.

    This reasoner applies the multi-layer belief model with:
    - Dempster–Shafer justification frame
    - minimal_dense_set as allocation function

    Only subsets with non-zero degree of belief are returned. The result
    is ordered first by increasing subset cardinality and, within each
    cardinality level, by decreasing degree of belief.

    Parameters
    ----------
    S : set 
        Set of possible states (frame of discernment).

    E : dict
        Quantitative evidence. Keys represent subsets of S (encoded using
        `get_key`), and values are belief masses associated with those
        subsets.

    Returns
    -------
    dict
        A dictionary mapping subsets of S (encoded using `get_key`) to
        their degree of belief. Only subsets with non-zero belief are
        included. The dictionary is ordered by increasing subset size and
        decreasing belief value.

    Example
    -------
    >>> S = {'dp', 'sp', 'dm'}
    >>> E = {get_key({'dp', 'dm'}): 0.8, get_key({'sp'}): 0.4}
    >>> result = ds_int_belief_model(S, E)
    >>> for k, b in result.items():
    ...     print(f"belief for {get_set(k)} -> {b:.2f}")
    >>> print("belief for other propositions -> 0")
    """
    model = MultiLayerBeliefModel(
        S,
        E,
        'dempster_shafer',
        minimal_dense_set
    )

    beliefs = []

    # Generate all subsets of S
    for n in range(1, len(S) + 1):  # start from 1 to skip empty set
        for subset in combinations(S, n):
            p = set(subset)
            b = model.degree_of_belief(p)

            if b != 0:
                k = get_key(p)
                beliefs.append((k, b))

    # Sort:
    # 1) by subset size (ascending)
    # 2) by belief value (descending)
    beliefs.sort(key=lambda x: (len(get_set(x[0])), -x[1]))

    # Build ordered dictionary
    result = {k: b for k, b in beliefs}

    return result


def sd_min_belief_model(S, E):
    """
    Compute degrees of belief for for non-empty subsets of S from evidence E.

    This reasoner applies the multi-layer belief model with:
    - strong denseness justification frame
    - minimal dense set as allocation function

    It is equivalent to applying the belief operator of topological evidence models to E when all pieces of evidence are assigned the same degree of certainty.

    Only subsets with non-zero degree of belief are returned. The result
    is ordered first by increasing subset cardinality and, within each
    cardinality level, by decreasing degree of belief.

    Parameters
    ----------
    S : set 
        Set of possible states (frame of discernment).

    E : dict
        Quantitative evidence. Keys represent subsets of S (encoded using
        `get_key`), and values are belief masses associated with those
        subsets.

    Returns
    -------
    dict
        A dictionary mapping subsets of S (encoded using `get_key`) to
        their degree of belief. Only subsets with non-zero belief are
        included. The dictionary is ordered by increasing subset size and
        decreasing belief value.

    Example
    -------
    >>> S = {'dp', 'sp', 'dm'}
    >>> E = {get_key({'dp', 'dm'}): 0.8, get_key({'sp'}): 0.4}
    >>> result = ds_int_belief_model(S, E)
    >>> for k, b in result.items():
    ...     print(f"belief for {get_set(k)} -> {b:.2f}")
    >>> print("belief for other propositions -> 0")
    """
    model = MultiLayerBeliefModel(
    S,
    E,
    'strong_denseness',
    minimal_dense_set
    )

    beliefs = []

    # Generate all subsets of S
    for n in range(1, len(S) + 1):  # start from 1 to skip empty set
        for subset in combinations(S, n):
            p = set(subset)
            b = model.degree_of_belief(p)

            if b != 0:
                k = get_key(p)
                beliefs.append((k, b))

    # Sort:
    # 1) by subset size (ascending)
    # 2) by belief value (descending)
    beliefs.sort(key=lambda x: (len(get_set(x[0])), -x[1]))

    # Build ordered dictionary
    result = {k: b for k, b in beliefs}

    return result

__all__ = ["ds_int_belief_model", "ds_min_belief_model", "sd_min_belief_model"]