from __future__ import annotations
import csv
from typing import Any, Dict, List, Optional, Tuple

from melody import Melody
from autocomplete_engines import LetterAutocompleteEngine, \
    SentenceAutocompleteEngine, MelodyAutocompleteEngine, sanitize
from test_compressed_prefix_tree import check_subtrees_non_increasing_order, \
    check_subtrees_compressibility, tree_weight_check
from test_simple_prefix_tree import scheck_subtrees_non_increasing_order, \
    stree_weight_check, scheck_subtrees_value
import random


def autocomplete_non_increasing_order(lst: List[Tuple]) -> bool:
    """Returns true if the autocomplete's output is sorted in
     non-increasing order"""
    for i in range(0, len(lst)-1):
        if lst[i][1] < lst[i+1][1]:
            return False
    return True


def num_duplicate_inputs(file: str, autocomplete_engine: str) -> Tuple(Dict[Any], int):
    """Returns a Tuple with a list of duplicate lines (after text sanitization) of the
    given file input and the total number of input values"""
    if autocomplete_engine == 'LetterAutocompleteEngine':
        input_set = set()
        duplicates = {}
        count = 0
        with open(file, encoding='utf8') as f:
            data = f.readlines()
            for line in data:
                sanitized = sanitize(line, 'letter')
                if all([x.isalnum() or x == ' ' for x in sanitized[0]]):
                    count += 1
                    if sanitized[0] in input_set:
                        if sanitized[0] in duplicates:
                            duplicates[sanitized[0]] += 1
                        else:
                            duplicates[sanitized[0]] = 1
                    else:
                        input_set.add(sanitized[0])
        return duplicates, count
    elif autocomplete_engine == 'SentenceAutocompleteEngine':
        input_set = set()
        duplicates = {}
        count = 0
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            # reader[x][0] is the line
            # reader[x][1] is the weight
            for row in reader:
                sanitized = sanitize(row[0], 'sentence')
                if len(sanitized[1]) >= 1 and \
                        any([x.isalnum() for x in sanitized[1]]):
                    count += 1
                    if sanitized[0] in input_set:
                        if sanitized[0] in duplicates:
                            duplicates[sanitized[0]] += 1
                        else:
                            duplicates[sanitized[0]] = 1
                    else:
                        input_set.add(sanitized[0])
        return duplicates, count


def sample_letter_autocomplete(data: Dict) -> LetterAutocompleteEngine:
    """A sample run of the letter autocomplete engine.

    data['file'] - file location
    data['autocompleter'] - 'simple' or 'compressed'
    data['weight_type'] - 'sum' or 'average'
    data['search'] - search input
    data['limit'] - autocomplete limit
    """
    engine = LetterAutocompleteEngine({
        'file': data['file'],
        'autocompleter': data['autocompleter'],
        'weight_type': data['weight_type']
    })
    # return engine.autocomplete(data['search'], data['limit'])
    return engine

def sample_sentence_autocomplete(data: Dict) -> List[Tuple[str, float]]:
    """A sample run of the sentence autocomplete engine.

    data['file'] - file location
    data['autocompleter'] - 'simple' or 'compressed'
    data['weight_type'] - 'sum' or 'average'
    data['search'] - search input
    data['limit'] - autocomplete limit
    """
    engine = SentenceAutocompleteEngine({
        'file': data['file'],
        'autocompleter': data['autocompleter'],
        'weight_type': data['weight_type']
    })
    return engine.autocomplete(data['search'], data['limit'])


def sample_melody_autocomplete(data: dict) -> None:
    """A sample run of the melody autocomplete engine.

    data['file'] - file location
    data['autocompleter'] - 'simple' or 'compressed'
    data['weight_type'] - 'sum' or 'average'
    data['search'] - search input
    data['limit'] - autocomplete limit
    """
    engine = MelodyAutocompleteEngine({
        'file': data['file'],
        'autocompleter': data['autocompleter'],
        'weight_type': data['weight_type']
    })
    melodies = engine.autocomplete(data['search'], data['limit'])
    for melody, _ in melodies:
        melody.play()


def test_sample_letter_autocomplete() -> None:
    """Tests
        1. CompressPrefixTree properties of autocompleter
            - compressibility check
            - subtrees non increasing order check
            - subtree weight check
        2. SimplePrefixTree properties of autocompleter
            - len(subtree.value) == len(spt.value) + 1 (when subtree is List)
            - subtrees non increasing order check
            - subtree weight check
        3. Test autocompleter properties
            - num_leaves == total_inputs - duplicate inputs
            - len(output) == limit
            - output weight is non-increasing
            - check leaves.weight == number of times it was inputted"""
    autocompleters = ['simple', 'compressed']
    files = ['data/lotr.txt', 'data/google_no_swears.txt']
    weight_types = ['sum', 'average']
    lotr_searches = ['l', 'fr', 'gan', 'pic', 'ring']
    google_searches = ['h', 'w', 'l', 'o', '']
    limits = [None] + random.sample(range(1, 100), 50)

    for file in files:
        for autocompleter in autocompleters:
            for weight_type in weight_types:
                # create engine
                engine = LetterAutocompleteEngine({
                    'file': file,
                    'autocompleter': autocompleter,
                    'weight_type': weight_type
                })

                if autocompleter == 'simple':
                    assert scheck_subtrees_non_increasing_order(engine.autocompleter)
                    assert scheck_subtrees_value(engine.autocompleter)
                    assert stree_weight_check(engine.autocompleter, weight_type)
                else:
                    # autocomplete == 'compressed'
                    assert check_subtrees_non_increasing_order(engine.autocompleter)
                    # if not check_subtrees_non_increasing_order(engine.autocompleter):
                    #     return engine.autocompleter
                    assert check_subtrees_compressibility(engine.autocompleter)
                    assert tree_weight_check(engine.autocompleter, weight_type)

                duplicates = num_duplicate_inputs(file, 'LetterAutocompleteEngine')
                total_duplicates = 0
                for duplicate in duplicates[0]:
                    total_duplicates += duplicates[0][duplicate]
                assert engine.autocompleter._num_leaves == duplicates[1] - total_duplicates

                if file == 'data/lotr.txt':
                    for search in lotr_searches:
                        for limit in limits:
                            output = engine.autocomplete(search, limit)
                            if limit is not None:
                                assert len(output) <= limit
                            assert autocomplete_non_increasing_order(output)
                else:
                    for search in google_searches:
                        for limit in limits:
                            output = engine.autocomplete(search, limit)
                            if limit is not None:
                                assert len(output) <= limit
                            assert autocomplete_non_increasing_order(output)

                # weights of duplicate entries must be greater than 1
                for duplicate in duplicates[0]:
                    output = engine.autocomplete(duplicate, 20)
                    for val in output:
                        if val == duplicate:
                            assert val[1] >= duplicates[0][duplicate]


def test_sample_sentence_autocomplete() -> None:
    """Tests
        1. CompressPrefixTree properties of autocompleter
            - compressibility check
            - subtrees non increasing order check
            - subtree weight check
        2. SimplePrefixTree properties of autocompleter
            - len(subtree.value) == len(spt.value) + 1 (when subtree is List)
            - subtrees non increasing order check
            - subtree weight check
        3. Test autocompleter properties
            - num_leaves == total_inputs - duplicate inputs
            - len(output) == limit
            - output weight is non-increasing
            - check leaves.weight == number of times it was inputted"""
    autocompleters = ['simple', 'compressed']
    files = ['data/google_searches.csv']
    weight_types = ['sum', 'average']
    google_searches = ['how', 'why', 'when', 'who', 'what']
    limits = [None] + random.sample(range(1, 200), 50)

    for file in files:
        for autocompleter in autocompleters:
            for weight_type in weight_types:
                # create engine
                engine = SentenceAutocompleteEngine({
                    'file': file,
                    'autocompleter': autocompleter,
                    'weight_type': weight_type
                })

                if autocompleter == 'simple':
                    assert scheck_subtrees_non_increasing_order(engine.autocompleter)
                    assert scheck_subtrees_value(engine.autocompleter)
                    assert stree_weight_check(engine.autocompleter, weight_type)
                else:
                    # autocomplete == 'compressed'
                    assert check_subtrees_non_increasing_order(engine.autocompleter)
                    assert check_subtrees_compressibility(engine.autocompleter)
                    assert tree_weight_check(engine.autocompleter, weight_type)

                duplicates = num_duplicate_inputs(file, 'SentenceAutocompleteEngine')
                total_duplicates = 0
                for duplicate in duplicates[0]:
                    total_duplicates += duplicates[0][duplicate]
                assert engine.autocompleter._num_leaves == duplicates[1] - total_duplicates

                for search in google_searches:
                    for limit in limits:
                        output = engine.autocomplete(search, limit)
                        if limit is not None:
                            assert len(output) <= limit
                        assert autocomplete_non_increasing_order(output)

                # weights of duplicate entries must be greater than 1
                for duplicate in duplicates[0]:
                    output = engine.autocomplete(duplicate, 20)
                    for val in output:
                        if val == duplicate:
                            assert val[1] >= duplicates[0][duplicate]


def test_sample_melody_autocomplete() -> None:
    """Tests
        1. CompressPrefixTree properties of autocompleter
            - compressibility check
            - subtrees non increasing order check
            - subtree weight check
        2. SimplePrefixTree properties of autocompleter
            - len(subtree.value) == len(spt.value) + 1 (when subtree is List)
            - subtrees non increasing order check
            - subtree weight check
        3. Test autocompleter properties
            - num_leaves == total_inputs - duplicate inputs
            - len(output) == limit
            - output weight is non-increasing
            - check leaves.weight == number of times it was inputted"""
    autocompleters = ['simple', 'compressed']
    files = ['data/random_melodies_c_scale.csv', 'data/songbook.csv']
    weight_types = ['sum', 'average']
    melody_searches = [[0, 1], [1, 2], [10], [0], [1], [2], [3], [4], [5], [6, 0]]
    limits = [None] + random.sample(range(1, 200), 50)

    for file in files:
        for autocompleter in autocompleters:
            for weight_type in weight_types:
                # create engine
                engine = MelodyAutocompleteEngine({
                    'file': file,
                    'autocompleter': autocompleter,
                    'weight_type': weight_type
                })

                if autocompleter == 'simple':
                    assert scheck_subtrees_non_increasing_order(engine.autocompleter)
                    assert scheck_subtrees_value(engine.autocompleter)
                    assert stree_weight_check(engine.autocompleter, weight_type)
                else:
                    # autocomplete == 'compressed'
                    assert check_subtrees_non_increasing_order(engine.autocompleter)
                    assert check_subtrees_compressibility(engine.autocompleter)
                    assert tree_weight_check(engine.autocompleter, weight_type)

                for search in melody_searches:
                    for limit in limits:
                        output = engine.autocomplete(search, limit)
                        if limit is not None:
                            assert len(output) <= limit
                        assert autocomplete_non_increasing_order(output)


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(5000)
    import pytest
    pytest.main(['test_autocomplete_engines.py'])
    # autocompleter = test_sample_letter_autocomplete()
    # check_subtrees_non_increasing_order(autocompleter)
