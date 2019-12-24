from hypothesis import given, settings
from hypothesis.strategies import integers
from prefix_tree import CompressedPrefixTree, SimplePrefixTree
from typing import Tuple, Any, List
import random


def check_subtrees_non_increasing_order(cpt: CompressedPrefixTree) -> bool:
    """Returns True if all subtrees are in non-increasing order
    If subtree.weight is negative return False
    """
    if isinstance(cpt.value, list):
        current_weight = -1
        for subtree in cpt.subtrees:
            if subtree.weight < 0:
                print('subtree value < 0')
                return False
            if current_weight < subtree.weight:
                if current_weight < 0:
                    current_weight = subtree.weight
                else:
                    return False
            current_weight = subtree.weight
            check_subtrees_non_increasing_order(subtree)
    return True


def check_subtrees_compressibility(cpt: CompressedPrefixTree) -> bool:
    """Return True if all internal values are incompressible

    >>> spt = SimplePrefixTree('sum')
    >>> spt.insert('lol', 1.0, ['l','o','l'])
    >>> check_subtrees_compressibility(spt)
    False
    >>> cpt = CompressedPrefixTree('sum')
    >>> cpt_subtree = CompressedPrefixTree('sum')
    >>> cpt_subtree.value = ['l','o','l']
    >>> cpt_sub_subtree = CompressedPrefixTree('sum')
    >>> cpt_sub_subtree.value = 'lol'
    >>> cpt_sub_subtree.weight = 1.0
    >>> cpt_subtree.subtrees.append(cpt_sub_subtree)
    >>> cpt_subtree.weight = 1.0
    >>> cpt.subtrees.append(cpt_subtree)
    >>> cpt.weight = 1.0
    >>> check_subtrees_compressibility(cpt)
    True
    """
    if isinstance(cpt.value, list):
        if cpt.value != []:
            # cpt.value != []
            if len(cpt.subtrees) == 1:
                if isinstance(cpt.subtrees[0].value, list):
                    # cpt is compressible
                    return False
        for subtree in cpt.subtrees:
            lst = []
            lst.append(check_subtrees_compressibility(subtree))
            return all(lst)
    return True


def tree_weight_check(cpt: CompressedPrefixTree, weight_type: str) -> bool:
    """Checks if the weight matches the attributes

    >>> cpt = CompressedPrefixTree('sum')
    >>> cpt.insert('lo', 10, ['l','o'])
    >>> tree_weight_check(cpt, 'sum')
    True
    >>> cpt.insert('lo', 10, ['l', 'o'])
    >>> tree_weight_check(cpt, 'sum')
    True
    >>> cpt.weight = 10
    >>> tree_weight_check(cpt, 'sum')
    False
    """
    if weight_type == 'sum':
        if cpt.weight != cpt.weight:
            return False
        weights = []
        for subtrees in cpt.subtrees:
            weights.append(tree_weight_check(subtrees, weight_type))
        return all(weights)
    else:
        if cpt.weight == 0 or len(cpt) == 0:
            return len(cpt) == 0 and cpt.total == 0
        elif cpt.weight != cpt.total/len(cpt):
            return False
        weights = []
        for subtrees in cpt.subtrees:
            weights.append(tree_weight_check(subtrees, weight_type))
        return all(weights)


def num_nodes(cpt: CompressedPrefixTree) -> int:
    """Return the number of nodes an SPT has

    >>> cpt = CompressedPrefixTree('sum')
    >>> cpt.insert('x', 1, ['x'])
    >>> num_nodes(cpt)
    2
    >>> cpt.insert('che', 3,['c','h','e'])
    >>> num_nodes(cpt)
    5
    >>> cpt.insert('xenon', 2, ['x','e','n','o','n'])
    >>> num_nodes(cpt)
    7
    """
    if cpt.weight == 0:
        return 1
    elif cpt.is_leaf():
        return 1
    else:
        # it's an internal node
        count = 1
        for subtree in cpt.subtrees:
            count += num_nodes(subtree)
        return count


def test_insert_2() -> None:
    """Test SimplePrefixTree.insert() method using different types of
    CPTs"""
    # sum
    cpt = CompressedPrefixTree('sum')
    # empty cpt
    assert len(cpt) == 0
    assert cpt.value == []
    # cpt w/ len == 1
    cpt.insert('x', 1, ['x'])
    assert len(cpt) == 1
    assert num_nodes(cpt) == 2
    # cpt w/ len == 1, internal nodes > 1, achieved in test_insert_num_nodes()
    # cpt w/ len == 2, internal nodes == 2
    cpt = CompressedPrefixTree('sum')
    cpt.insert('x', 1, [])
    assert len(cpt) == 1
    assert num_nodes(cpt) == 2


def cpt_method_3(cpt: CompressedPrefixTree, largest_prefix: int, weights: List[Any],
                 prefixes: List[Any] = [[]]) -> List[List[Any]]:
    """Create a specialized generated spt for testing purposes

                        []
                [0]           [1]
            [0,0] [0,1]   [1,0] [1,1]
            ...             ...

    Note: height of spt = len(largest prefix tree) + 2 = largest_prefix + 2
    """
    if not isinstance(cpt.value, list):
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
                cpt.insert(values.pop(), float(weights.pop()), pref)
                accum_prefixes.extend(cpt_method_3(cpt, largest_prefix, weights, [pref]))
        return accum_prefixes


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
        cpt = CompressedPrefixTree('sum')
        cpt_avg = CompressedPrefixTree('average')
        if method == '3':
            prefixes = cpt_method_3(cpt, 3, list(range(15)))
            cpt_method_3(cpt_avg, 3, list(range(15)))
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
                cpt.insert(values[len(values)-1], weights[len(weights)-1], prefixes[len(prefixes)-1])
                cpt_avg.insert(values[len(values) - 1], weights[len(weights) - 1], prefixes[len(prefixes) - 1])
        if method == '1':
            assert len(cpt.subtrees) == len(prefixes)
        elif method == '2':
            assert len(cpt.subtrees) == 2
        else:   # method == '3'
            assert len(cpt.subtrees) == 2
        assert cpt.weight == sum(weights)
        assert cpt_avg.weight == sum(weights)/len(values)
        assert len(cpt) == len(values)
        assert check_subtrees_non_increasing_order(cpt)
        assert check_subtrees_non_increasing_order(cpt_avg)
        assert check_subtrees_compressibility(cpt)
        assert tree_weight_check(cpt, 'sum')
        assert tree_weight_check(cpt_avg, 'average')


@given(length=integers(min_value=10, max_value=100))
def test_remove(length: int) -> None:
    """Test remove method in the SimplePrefixTree class"""
    methods = ['1', '2', '3']

    for method in methods:
        prefixes = []
        values = []
        weights = []
        cpt = CompressedPrefixTree('sum')
        cpt_avg = CompressedPrefixTree('average')

        if method == '3':
            prefixes = cpt_method_3(cpt, 3, list(range(15)))
            cpt_method_3(cpt_avg, 3, list(range(15)))
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
                cpt.insert(values[len(values) - 1], weights[len(weights) - 1],
                           prefixes[len(prefixes) - 1])
                cpt_avg.insert(values[len(values) - 1], weights[len(weights) - 1],
                               prefixes[len(prefixes) - 1])
        if method == '1':
            for prefix in prefixes:
                prev_weight = cpt.weight
                prev_weight_avg = cpt_avg.weight
                prev_sum = cpt_avg.total
                prev_num = len(cpt)
                prev_num_nodes = num_nodes(cpt)
                cpt.remove(prefix)
                cpt_avg.remove(prefix)
                assert len(cpt) < prev_num   # deleting at least 1 leaf
                assert cpt_avg.total < prev_sum
                if len(cpt) == 0:
                    assert cpt.weight == 0
                else:
                    assert cpt_avg.weight == (cpt_avg.total/len(cpt))
                assert prev_weight_avg == (prev_sum/prev_num)
                assert cpt.weight == cpt_avg.total
                assert cpt.weight < prev_weight == prev_sum    # weight_type: 'sum'
                assert num_nodes(cpt) < prev_num_nodes
                assert check_subtrees_non_increasing_order(cpt)
                assert check_subtrees_non_increasing_order(cpt_avg)
                assert check_subtrees_compressibility(cpt)
                assert tree_weight_check(cpt, 'sum')
                assert tree_weight_check(cpt_avg, 'average')
        elif method == '2':
            for prefix in prefixes:
                prev_weight = cpt.weight
                prev_weight_avg = cpt_avg.weight
                prev_sum = cpt_avg.total
                prev_num = len(cpt_avg)
                prev_num_nodes = num_nodes(cpt)
                cpt_avg.remove(prefix)
                cpt.remove(prefix)
                assert len(cpt_avg) < prev_num  # deleting 1 leaf
                assert len(cpt) == prev_num - 1
                assert cpt_avg.total < prev_sum
                if len(cpt) == 0:
                    assert cpt.weight == 0
                else:
                    assert cpt_avg.weight == (cpt_avg.total / len(cpt))
                assert prev_weight_avg == (prev_sum / prev_num)
                assert cpt.weight == cpt_avg.total
                assert cpt.weight < prev_weight == prev_sum  # weight_type: 'sum'
                assert num_nodes(cpt) < prev_num_nodes
                assert check_subtrees_non_increasing_order(cpt)
                assert check_subtrees_non_increasing_order(cpt_avg)
                assert check_subtrees_compressibility(cpt)
                assert tree_weight_check(cpt, 'sum')
                assert tree_weight_check(cpt_avg, 'average')
        elif method == '3':
            prefixes.reverse()
            for prefix in prefixes:
                prev_weight = cpt.weight
                prev_weight_avg = cpt_avg.weight
                prev_sum = cpt_avg.total
                prev_num = len(cpt)
                prev_num_nodes = num_nodes(cpt)
                cpt.remove(prefix)
                cpt_avg.remove(prefix)
                assert len(cpt) < prev_num  # deleting 1 leaf
                assert len(cpt) == prev_num - 1
                assert cpt_avg.total < prev_sum
                if len(cpt) == 0:
                    assert cpt.weight == 0
                else:
                    assert cpt_avg.weight == (cpt.weight / len(cpt))
                assert prev_weight_avg == (prev_sum / prev_num)
                assert cpt.weight == cpt_avg.total
                assert cpt.weight < prev_weight == prev_sum  # weight_type: 'sum'
                assert num_nodes(cpt) < prev_num_nodes
                assert check_subtrees_non_increasing_order(cpt)
                assert check_subtrees_non_increasing_order(cpt_avg)
                assert check_subtrees_compressibility(cpt)
                assert tree_weight_check(cpt, 'sum')
                assert tree_weight_check(cpt_avg, 'average')
            prefixes.reverse()


@given(length=integers(min_value=10, max_value=100))
def test_autocomplete(length: int) -> None:
    """Test the aggregate weight, length,..etc of the CompressedPrefixTree"""
    import sys
    sys.setrecursionlimit(5000)

    # insertion method 1 (n = length)
    # prefixes = [[0,..,n-1],[1,..,n-1],[2,...n-1],....[n-1]]
    # cpt must len(prefixes) subtrees

    # insertion method 2 (n = length)
    # prefixes = [[0,..,n-1],[0,..,n-2],[0,...n-3],....[1]]
    # cpt must have 1 subtree

    # insertion method 3 (n = length)
    # check method_cpt3()
    methods = ['1', '2', '3']

    for method in methods:
        prefixes = []
        values = []
        weights = []
        cpt = CompressedPrefixTree('sum')
        cpt_avg = CompressedPrefixTree('average')

        if method == '3':
            prefixes = cpt_method_3(cpt, 3, list(range(15)))
            cpt_method_3(cpt_avg, 3, list(range(15)))
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
                cpt.insert(values[len(values) - 1], weights[len(weights) - 1],
                           prefixes[len(prefixes) - 1])
                cpt_avg.insert(values[len(values) - 1], weights[len(weights) - 1],
                               prefixes[len(prefixes) - 1])

        prefixes.insert(0, [])
        for prefix in prefixes:
            for i in range(1, len(values) + 1):
                assert len(cpt.autocomplete(prefix, i)) <= i
                assert len(cpt.autocomplete(prefix, i ** 2)) <= len(
                    values)
                assert len(cpt_avg.autocomplete(prefix, i)) <= i
                assert len(cpt_avg.autocomplete(prefix, i ** 2)) <= len(
                    values)
                tup = cpt.autocomplete(prefix, i)
                tup_av = cpt_avg.autocomplete(prefix, i)
                for x in range(len(tup)):
                    # weights[0] should have the greatest weight
                    assert tup[x][1] <= weights[0]
                    assert tup_av[x][1] <= weights[0]
                    if x != len(tup) - 1:
                        # weights should be non-increasing
                        assert tup[x][1] >= tup[x + 1][1]
                        assert tup_av[x][1] >= tup[x + 1][1]
        prefixes.pop(0)     # popping [] out


def test_cpt_rep_invariant() -> None:
    """Tests cpt representation invariant specifically for the case where
    the root isn't == []"""

    for weight in ['average']:

        cpt = CompressedPrefixTree(weight)
        cpt.insert('doggy', 1.0, ['d', 'o', 'g', 'g', 'y'])
        assert cpt.value == ['d', 'o', 'g', 'g', 'y']

        cpt.insert('donna', 2.0, ['d', 'o', 'n', 'n', 'a'])
        assert cpt.value == ['d', 'o']
        assert cpt.subtrees[0].value == ['d', 'o', 'n', 'n', 'a']

        cpt.insert('dogi', 2.0, ['d', 'o', 'g', 'i'])
        assert cpt.value == ['d', 'o']
        assert cpt.subtrees[1].value == ['d', 'o', 'g']
        assert cpt.subtrees[1].subtrees[0].value == ['d', 'o', 'g', 'i']
        assert cpt.subtrees[1].subtrees[1].value == ['d', 'o', 'g', 'g', 'y']

        cpt.insert('dim', 5.0, ['d', 'i', 'm'])
        assert cpt.value == ['d']
        assert cpt.subtrees[0].value == ['d', 'i', 'm']
        assert cpt.subtrees[1].subtrees[1].value == ['d', 'o', 'g']

        cpt.insert('che', 10.0, ['c', 'h', 'e'])
        assert cpt.value == []
        assert cpt.subtrees[0].value == ['c', 'h', 'e']
        assert cpt.subtrees[1].value == ['d']

        cpt.remove(['c'])
        assert cpt.value == ['d']
        assert cpt.subtrees[0].value == ['d', 'i', 'm']
        assert cpt.subtrees[1].subtrees[1].value == ['d', 'o', 'g']

        cpt.remove(['d', 'o', 'g'])
        assert cpt.value == ['d']
        assert cpt.subtrees[0].value == ['d', 'i', 'm']
        assert cpt.subtrees[1].value == ['d', 'o', 'n', 'n', 'a']

        cpt.remove(['d', 'i'])
        assert cpt.value == ['d', 'o', 'n', 'n', 'a']
        assert len(cpt.subtrees) == 1
        assert cpt.subtrees[0].is_leaf

        cpt.remove(['d'])
        assert cpt.value == []
        assert len(cpt.subtrees) == 0
        assert cpt.weight == 0

if __name__ == '__main__':
    import pytest
    pytest.main(['test_compressed_prefix_tree.py'])
