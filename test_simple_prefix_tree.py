from hypothesis import given, settings
from hypothesis.strategies import integers
from prefix_tree import SimplePrefixTree
from typing import Tuple, Any, List
import random


def stree_weight_check(spt: SimplePrefixTree, weight_type: str) -> bool:
    """Checks if the weight matches the attributes

    >>> cpt = SimplePrefixTree('sum')
    >>> cpt.insert('lo', 10, ['l','o'])
    >>> stree_weight_check(cpt, 'sum')
    True
    >>> cpt.insert('lo', 10, ['l', 'o'])
    >>> stree_weight_check(cpt, 'sum')
    True
    >>> cpt.weight = 10
    >>> stree_weight_check(cpt, 'sum')
    False
    """
    if weight_type == 'sum':
        if spt.weight != len(spt)*spt.weight:
            return False
        weights = []
        for subtrees in spt.subtrees:
            weights.append(stree_weight_check(subtrees, weight_type))
        return all(weights)
    else:
        if spt.weight == 0:
            return len(spt) == 0 and len(spt)*spt.weight == 0
        elif spt.weight != len(spt)*spt.weight/len(spt):
            return False
        weights = []
        for subtrees in spt.subtrees:
            weights.append(stree_weight_check(subtrees, weight_type))
        return all(weights)


def scheck_subtrees_non_increasing_order(spt: SimplePrefixTree) -> bool:
    """Returns True if all subtrees are in non-increasing order
    If subtree.weight is negative return False
    """
    if isinstance(spt.value, list):
        current_weight = -1
        for subtree in spt.subtrees:
            if subtree.weight < 0:
                print('subtree value < 0')
                return False
            if current_weight < subtree.weight:
                if current_weight < 0:
                    current_weight = subtree.weight
                else:
                    return False
            current_weight = subtree.weight
            scheck_subtrees_non_increasing_order(subtree)
    return True


def scheck_subtrees_value(spt: SimplePrefixTree) -> bool:
    """Return True if (subtree.value is List -> len(subtree.value) - 1
    == len(self.value). Checks if representation variant is held

    >>> spt = SimplePrefixTree('sum')
    >>> scheck_subtrees_value(spt)
    True
    >>> spt.insert('loling', 1.0, ['l','o','l','i','n','g'])
    >>> scheck_subtrees_value(spt)
    True
    >>> spt = SimplePrefixTree('sum')
    >>> spt_subtree = SimplePrefixTree('sum')
    >>> spt_subtree.insert('mzxcvb', 1.0, ['m','z','x','c','v','b'])
    >>> spt.weight = 1.0
    >>> spt.subtrees.append(spt_subtree)
    >>> scheck_subtrees_value(spt)
    False
    """
    if isinstance(spt.value, list):
        for subtree in spt.subtrees:
            if isinstance(subtree.value, list) and \
                    len(subtree.value) != len(spt.value) + 1:
                    return False
            scheck_subtrees_value(subtree)
    return True


def spt_height(spt: SimplePrefixTree) -> int:
    """Return the height of the spt

    Precondition: spt is not an empty SimplePrefixTree
    """
    if spt.is_leaf():  # an internal node with only 1 leaf
        return 1
    else:
        # spt is not a leaf
        count = 1
        for subtree in spt.subtrees:
            height = spt_height(subtree)
            if height >= count:
                count = height + 1  # + 1 including the own tree
        return count


# def left_leaf(spt: SimplePrefixTree) -> Tuple[Any, float]:
#     """Return the left most leaf in a SimplePrefixTree as (value, weight)
#
#     Precondition:
#         - spt is not an empty SimplePrefixTree
#     """
#     if spt.weight == 0:
#         return [], 0
#     elif spt.weight > 0 and spt.subtrees == []:
#         return spt.value, spt.weight
#     else:
#         return left_leaf(spt.subtrees[0])


# def right_leaf(spt: SimplePrefixTree) -> Tuple[Any, float]:
#     """Return the right most leaf in a SimplePrefixTree as (value, weight)
#
#     Precondition:
#         - spt is not an empty SimplePrefixTree
#     """
#     if spt.weight == 0:
#         return 0, 0
#     elif spt.weight > 0 and spt.subtrees == []:
#         return spt.value, spt.weight
#     else:
#         return right_leaf(spt.subtrees[len(spt.subtrees)-1])


def num_nodes(spt: SimplePrefixTree) -> int:
    """Return the number of nodes an SPT has

    >>> spt = SimplePrefixTree('sum')
    >>> spt.insert('x', 1, ['x'])
    >>> num_nodes(spt)
    3
    >>> spt.insert('che', 3,['c','h','e'])
    >>> num_nodes(spt)
    7
    >>> spt.insert('xenon', 2, ['x','e','n','o','n'])
    >>> num_nodes(spt)
    12
    """
    if spt.weight == 0:
        return 1
    elif spt.is_leaf():
        return 1
    else:
        # it's an internal node
        count = 1
        for subtree in spt.subtrees:
            count += num_nodes(subtree)
        return count


def spt_method_3(spt: SimplePrefixTree, largest_prefix: int, weights: List[Any],
                 prefixes: List[Any] = [[]]) -> List[List[Any]]:
    """Create a specialized generated spt for testing purposes

                        []
                [0]           [1]
            [0,0] [0,1]   [1,0] [1,1]
            ...             ...

    Note: height of spt = len(largest prefix tree) + 2 = largest_prefix + 2
    """
    if not isinstance(spt.value, list):
        return []
    elif len(prefixes[0]) == largest_prefix:
        return []
    else:
        # extract the prefix
        accum_prefixes = []
        # values don't matter
        values = random.sample(range(1000000), 10)
        for prefix in prefixes:
            for n in range(0, 2):
                pref = prefix + [n]
                accum_prefixes.append(pref)
                spt.insert(values.pop(), float(weights.pop()), pref)
                accum_prefixes.extend(spt_method_3(spt, largest_prefix, weights, [pref]))
        return accum_prefixes


@given(length=integers(min_value=10, max_value=1000))
def test_insert_num_nodes(length: int) -> None:
    """Inserting one value with a length-n prefix [x_1, .., x_n] into a new
    prefix tree should result in a tree with (n+2) nodes. (n+1) internal nodes
    plus 1 inserted value"""
    import sys
    sys.setrecursionlimit(5000)

    prefix = list(range(length))
    spt = SimplePrefixTree('sum')
    spt.insert('x', 1, prefix)
    assert num_nodes(spt) == (length + 2)
    assert len(spt) == 1
    assert spt.weight == 1


@given(length=integers(min_value=10, max_value=100))
def test_insert(length: int) -> None:
    """Test the aggregate weight, length,..etc of the SimplePrefixTree"""
    import sys
    sys.setrecursionlimit(5000)

    # insertion method 1 (n = length)
    # prefixes = [[0,..,n-1],[1,..,n-1],[2,...n-1],....[n-1]]
    # spt must len(prefixes) subtrees

    # insertion method 2 (n = length)
    # prefixes = [[0,..,n-1],[0,..,n-2],[0,...n-3],....[0]]
    # spt must have 1 subtree

    # insertion method 3 (n = length)

    methods = ['1', '2', '3']

    for method in methods:
        prefixes = []
        values = []
        weights = []
        spt = SimplePrefixTree('sum')
        spt_avg = SimplePrefixTree('average')
        if method == '3':
            prefixes = spt_method_3(spt, 3, list(range(15)))
            spt_method_3(spt_avg, 3, list(range(15)))
            values = prefixes   # values is only tested on length
            weights = list(range(15))
            weights.reverse()
        else:
            for x in range(0, length):
                if method == '1':
                    start = x
                    stop = length
                else:
                    start = 0
                    stop = length - x
                prefixes.append(list(range(start, stop)))
                values.append(length - x)
                weights.append(length - x)
                spt.insert(values[len(values)-1], weights[len(weights)-1], prefixes[len(prefixes)-1])
                spt_avg.insert(values[len(values) - 1], weights[len(weights) - 1], prefixes[len(prefixes) - 1])
        if method == '1':
            assert len(spt.subtrees) == len(prefixes)
        elif method == '2':
            assert len(spt.subtrees) == 1
        else:   # method == '3'
            assert len(spt.subtrees) == 2
        assert spt.weight == sum(weights)
        assert spt_avg.weight == sum(weights)/len(values)
        assert len(spt) == len(values)
        # check if spt has non-increasing weight order
        assert scheck_subtrees_non_increasing_order(spt)
        #assert stree_weight_check(spt, 'sum')
        assert stree_weight_check(spt_avg, 'average')

def test_insert_2() -> None:
    """Test SimplePrefixTree.insert() method using different types of
    SPTs"""
    # sum
    spt = SimplePrefixTree('sum')
    # empty spt
    assert len(spt) == 0
    assert spt.value == []
    # spt w/ len == 1
    spt.insert('x', 1, ['x'])
    assert len(spt) == 1
    assert num_nodes(spt) == 3
    # spt w/ len == 1, internal nodes > 1, achieved in test_insert_num_nodes()
    # spt w/ len == 2, internal nodes == 2
    spt = SimplePrefixTree('sum')
    spt.insert('x', 1, [])
    assert len(spt) == 1
    assert num_nodes(spt) == 2


@given(length=integers(min_value=10, max_value=100))
def test_autocomplete(length: int) -> None:
    """Test the aggregate weight, length,..etc of the SimplePrefixTree"""
    import sys
    sys.setrecursionlimit(5000)

    # insertion method 1 (n = length)
    # prefixes = [[0,..,n-1],[1,..,n-1],[2,...n-1],....[n-1]]
    # every prefix has 1 value
    # spt will have 'n' subtrees
    # spt must len(prefixes) subtrees

    # insertion method 2 (n = length)
    # prefixes = [[0,..,n-1],[0,..,n-2],[0,...n-3],....[1]]
    # every prefix has 1 value
    # spt must have 1 subtree

    # insertion method 3 (n = length)
    # check method_spt3()
    # balanced spt

    methods = ['1', '2', '3']

    for method in methods:
        prefixes = []
        values = []
        weights = []
        spt = SimplePrefixTree('sum')
        spt_avg = SimplePrefixTree('average')

        if method == '3':
            prefixes = spt_method_3(spt, 3, list(range(15)))
            spt_method_3(spt_avg, 3, list(range(15)))
            values = prefixes   # values is only tested on length
            weights = list(range(15))
            weights.reverse()
        else:
            for x in range(0, length):
                if method == '1':
                    start = x
                    stop = length
                else:
                    start = 0
                    stop = length - x
                prefixes.append(list(range(start, stop)))
                values.append(length - x)
                # weight goes for values, go from weight = length, to weight = 1
                weights.append(length - x)
                spt.insert(values[len(values) - 1], weights[len(weights) - 1],
                           prefixes[len(prefixes) - 1])
                spt_avg.insert(values[len(values) - 1], weights[len(weights) - 1],
                               prefixes[len(prefixes) - 1])

        prefixes.insert(0, [])
        for prefix in prefixes:
            for i in range(1, len(values) + 1):
                assert len(spt.autocomplete(prefix, i)) <= i
                assert len(spt.autocomplete(prefix, i ** 2)) <= len(
                    values)
                assert len(spt_avg.autocomplete(prefix, i)) <= i
                assert len(spt_avg.autocomplete(prefix, i ** 2)) <= len(
                    values)
                tup = spt.autocomplete(prefix, i)
                tup_av = spt_avg.autocomplete(prefix, i)
                for x in range(len(tup)):
                    # weights[0] should have the greatest weight
                    assert tup[x][1] <= weights[0]
                    assert tup_av[x][1] <= weights[0]
                    if x != len(tup) - 1:
                        # weights should be non-increasing
                        assert tup[x][1] >= tup[x + 1][1]
                        assert tup_av[x][1] >= tup[x + 1][1]
        prefixes.pop(0)     # popping [] out


@given(length=integers(min_value=10, max_value=100))
def test_remove(length: int) -> None:
    """Test remove method in the SimplePrefixTree class"""
    methods = ['1', '2', '3']

    for method in methods:
        prefixes = []
        values = []
        weights = []
        spt = SimplePrefixTree('sum')
        spt_avg = SimplePrefixTree('average')

        if method == '3':
            prefixes = spt_method_3(spt, 3, list(range(15)))
            spt_method_3(spt_avg, 3, list(range(15)))
            values = prefixes   # values is only tested on length
            weights = list(range(15))
            weights.reverse()
        else:
            for x in range(0, length):
                if method == '1':
                    start = x
                    stop = length
                elif method == '2':
                    start = 0
                    stop = length - x
                prefixes.append(list(range(start, stop)))
                values.append(length - x)
                # weight goes for values, go from weight = length, to weight = 1
                weights.append(length - x)
                spt.insert(values[len(values) - 1], weights[len(weights) - 1],
                           prefixes[len(prefixes) - 1])
                spt_avg.insert(values[len(values) - 1], weights[len(weights) - 1],
                               prefixes[len(prefixes) - 1])
        if method == '1':
            for prefix in prefixes:
                prev_weight = spt.weight
                prev_weight_avg = spt_avg.weight
                prev_sum = spt.weight
                prev_num = len(spt)
                prev_num_nodes = num_nodes(spt)
                spt.remove([prefix[0]])
                spt_avg.remove([prefix[0]])
                assert len(spt) < prev_num   # deleting at least 1 leaf
                assert spt.weight < prev_sum
                if len(spt) == 0:
                    assert spt.weight == 0
                else:
                    assert spt_avg.weight == (spt.weight/len(spt))
                assert prev_weight_avg == (prev_sum/prev_num)
                assert spt.weight == len(spt)*spt_avg.weight
                assert spt.weight < prev_weight == prev_sum    # weight_type: 'sum'
                assert num_nodes(spt) < prev_num_nodes
                assert scheck_subtrees_non_increasing_order(spt)
                assert scheck_subtrees_non_increasing_order(spt_avg)
                assert scheck_subtrees_value(spt)
                assert scheck_subtrees_value(spt_avg)
                #assert stree_weight_check(spt, 'sum')
                assert stree_weight_check(spt_avg, 'average')
        elif method == '2':
            for prefix in prefixes:
                prev_weight = spt.weight
                prev_weight_avg = spt_avg.weight
                prev_sum = len(spt)*spt.weight
                prev_num = len(spt)
                prev_num_nodes = num_nodes(spt)
                spt.remove(prefix)
                spt_avg.remove(prefix)
                assert len(spt) < prev_num  # deleting 1 leaf
                assert len(spt) == prev_num - 1
                assert len(spt)*spt.weight < prev_sum
                if len(spt) == 0:
                    assert spt.weight == 0
                else:
                    assert spt_avg.weight == (len(spt)*spt.weight / len(spt))
                assert prev_weight_avg == (prev_sum / prev_num)
                assert spt.weight == len(spt)*spt.weight
                assert spt.weight < prev_weight == prev_sum  # weight_type: 'sum'
                assert prev_num_nodes - num_nodes(spt) == 2
                assert scheck_subtrees_non_increasing_order(spt)
                assert scheck_subtrees_non_increasing_order(spt_avg)
                assert scheck_subtrees_value(spt)
                assert scheck_subtrees_value(spt_avg)
                assert stree_weight_check(spt, 'sum')
                assert stree_weight_check(spt_avg, 'average')
        elif method == '3':
            prefixes.reverse()
            for prefix in prefixes:
                prev_weight = spt.weight
                prev_weight_avg = spt_avg.weight
                prev_sum = len(spt)*spt.weight
                prev_num = len(spt)
                prev_num_nodes = num_nodes(spt)
                spt.remove(prefix)
                spt_avg.remove(prefix)
                assert len(spt) < prev_num  # deleting 1 leaf
                assert len(spt) == prev_num - 1
                assert len(spt)*spt.weight < prev_sum
                if len(spt) == 0:
                    assert spt.weight == 0
                else:
                    assert spt_avg.weight == (len(spt)*spt.weight / len(spt))
                assert prev_weight_avg == (prev_sum / prev_num)
                assert spt.weight == len(spt)*spt.weight
                assert spt.weight < prev_weight == prev_sum  # weight_type: 'sum'
                assert prev_num_nodes - num_nodes(spt) == 2
                assert scheck_subtrees_non_increasing_order(spt)
                assert scheck_subtrees_non_increasing_order(spt_avg)
                assert scheck_subtrees_value(spt)
                assert scheck_subtrees_value(spt_avg)
                assert stree_weight_check(spt, 'sum')
                assert stree_weight_check(spt_avg, 'average')
            prefixes.reverse()


if __name__ == '__main__':
    import pytest
    pytest.main(['test_simple_prefix_tree.py'])

