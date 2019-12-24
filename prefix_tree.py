"""CSC148 Assignment 2: Autocompleter classes

=== CSC148 Fall 2018 ===
Department of Computer Science,
University of Toronto

=== Module Description ===
This file contains the design of a public interface (Autocompleter) and two
implementation of this interface, SimplePrefixTree and CompressedPrefixTree.
You'll complete both of these subclasses over the course of this assignment.

As usual, be sure not to change any parts of the given *public interface* in the
starter code---and this includes the instance attributes, which we will be
testing directly! You may, however, add new private attributes, methods, and
top-level functions to this file.
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple


################################################################################
# The Autocompleter ADT
################################################################################
class Autocompleter:
    """An abstract class representing the Autocompleter Abstract Data Type.
    """

    def __len__(self) -> int:
        """Return the number of values stored in this Autocompleter."""
        raise NotImplementedError

    def insert(self, value: Any, weight: float, prefix: List) -> None:
        """Insert the given value into this Autocompleter.

        The value is inserted with the given weight, and is associated with
        the prefix sequence <prefix>.

        If the value has already been inserted into this prefix tree
        (compare values using ==), then the given weight should be *added* to
        the existing weight of this value.

        Preconditions:
            weight > 0
            The given value is either:
                1) not in this Autocompleter
                2) was previously inserted with the SAME prefix sequence
        """
        raise NotImplementedError

    def autocomplete(self, prefix: List,
                     limit: Optional[int] = None) -> List[Tuple[Any, float]]:
        """Return up to <limit> matches for the given prefix.

        The return value is a list of tuples (value, weight), and must be
        ordered in non-increasing weight. (You can decide how to break ties.)

        If limit is None, return *every* match for the given prefix.

        Precondition: limit is None or limit > 0.
        """
        raise NotImplementedError

    def remove(self, prefix: List) -> None:
        """Remove all values that match the given prefix.
        """
        raise NotImplementedError


################################################################################
# SimplePrefixTree (Tasks 1-3)
################################################################################
class SimplePrefixTree(Autocompleter):
    """A simple prefix tree.

    This class follows the implementation described on the assignment handout.
    Note that we've made the attributes public because we will be accessing them
    directly for testing purposes.

    === Attributes ===
    value:
        The value stored at the root of this prefix tree, or [] if this
        prefix tree is empty.
    weight:
        The weight of this prefix tree. If this tree is a leaf, this attribute
        stores the weight of the value stored in the leaf. If this tree is
        not a leaf and non-empty, this attribute stores the *aggregate weight*
        of the leaf weights in this tree.
    subtrees:
        A list of subtrees of this prefix tree.
    weight_type:
        The way of measure for weight being use, either sum of all leaf values
        or the average of all leaf values

    === Representation invariants ===
    - self.weight >= 0

    - (EMPTY TREE):
        If self.weight == 0, then self.value == [] and self.subtrees == [].
        This represents an empty simple prefix tree.
    - (LEAF):
        If self.subtrees == [] and self.weight > 0, this tree is a leaf.
        (self.value is a value that was inserted into this tree.)
    - (NON-EMPTY, NON-LEAF):
        If len(self.subtrees) > 0, then self.value is a list (*common prefix*),
        and self.weight > 0 (*aggregate weight*).

    - ("prefixes grow by 1")
      If len(self.subtrees) > 0, and subtree in self.subtrees, and subtree
      is non-empty and not a leaf, then

          subtree.value == self.value + [x], for some element x

    - self.subtrees does not contain any empty prefix trees.
    - self.subtrees is *sorted* in non-increasing order of their weights.
      (You can break ties any way you like.)
      Note that this applies to both leaves and non-leaf subtrees:
      both can appear in the same self.subtrees list, and both have a `weight`
      attribute.
    """
    value: Any
    weight: float
    subtrees: List[SimplePrefixTree]
    weight_type: str

    def __init__(self, weight_type: str) -> None:
        """Initialize an empty simple prefix tree.

        Precondition: weight_type == 'sum' or weight_type == 'average'.

        The given <weight_type> value specifies how the aggregate weight
        of non-leaf trees should be calculated (see the assignment handout
        for details).
        """
        self.value = []
        self.subtrees = []
        self.weight = 0.0
        self.weight_type = weight_type

    def is_empty(self) -> bool:
        """Return whether this simple prefix tree is empty."""
        return self.weight == 0.0

    def is_leaf(self) -> bool:
        """Return whether this simple prefix tree is a leaf."""
        return self.weight > 0 and self.subtrees == []

    def __str__(self) -> str:
        """Return a string representation of this tree.

        You may find this method helpful for debugging.
        """
        return self._str_indented()

    def _str_indented(self, depth: int = 0) -> str:
        """Return an indented string representation of this tree.

        The indentation level is specified by the <depth> parameter.
        """
        if self.is_empty():
            return ''
        else:
            s = '  ' * depth + f'{self.value} ({self.weight})\n'
            for subtree in self.subtrees:
                s += subtree._str_indented(depth + 1)
            return s

    def __len__(self) -> int:
        """ Refer to Parent Class
        """
        size = 0
        for subtree in self.subtrees:
            if subtree.is_leaf():
                size += 1
            else:
                size += len(subtree)
        return size

    def insert(self, value: Any, weight: float, prefix: List) -> None:
        """ Refer to Parent Class
        """
        if self.weight_type == 'sum':
            self.weight += weight
        else:
            if self._insertion_repeat_indicator(value, prefix):
                self.weight = (self.weight * len(self) + weight) / len(self)
            else:
                self.weight = (self.weight * len(self) + weight) / (
                    len(self) + 1)
        if len(prefix) == 0:
            indicate = 0
            for element in self.subtrees:
                if value == element.value:
                    element.weight += weight
                    indicate += 1
                    self.subtrees.sort(key=lambda x: x.weight, reverse=True)
            if indicate == 0:
                new_item = SimplePrefixTree(self.weight_type)
                new_item.value = value
                new_item.weight = weight
                self.subtrees.append(new_item)
                self.subtrees.sort(key=lambda x: x.weight, reverse=True)
        else:
            pref = SimplePrefixTree(self.weight_type)
            pref.value = self.value + [prefix[0]]
            indicate = 0
            for element in self.subtrees:
                if pref.value == element.value:
                    index = self.subtrees.index(element)
                    self.subtrees[index].insert(value, weight, prefix[1:])
                    indicate += 1
                    self.subtrees.sort(key=lambda x: x.weight, reverse=True)
            if indicate == 0:
                self.subtrees.append(pref)
                index = self.subtrees.index(pref)
                self.subtrees[index].insert(value, weight, prefix[1:])
                self.subtrees.sort(key=lambda x: x.weight, reverse=True)

    def _insertion_repeat_indicator(self, value: Any, prefix: List) -> bool:
        """ Indicate whether the insert value is already inserted
        """
        if len(prefix) == 0:
            for subtree in self.subtrees:
                if subtree.value == value:
                    return True
        else:
            for subtree in self.subtrees:
                if subtree.value[len(subtree.value) - 1] == prefix[0]:
                    return subtree._insertion_repeat_indicator(value, prefix[1:])
        return False

    def autocomplete(self, prefix: List,
                     limit: Optional[int] = None) -> List[Tuple[Any, float]]:
        """ Refer to Parent Class
        """
        if len(prefix) == 0:
            newlist = []
            for subtree in self.subtrees:
                if limit is None or len(newlist) < limit:
                    if subtree.is_leaf():
                        newlist = newlist + [(subtree.value, subtree.weight)]
                        newlist.sort(key=lambda x: x[1], reverse=True)
                    else:
                        result = subtree.autocomplete([], limit)
                        newlist.extend(result)
                        newlist.sort(key=lambda x: x[1], reverse=True)
            return newlist[:limit]
        else:
            pref = self.value + [prefix[0]]
            for element in self.subtrees:
                if pref == element.value:
                    return element.autocomplete(prefix[1:], limit)
        return []

    def remove(self, prefix: List) -> None:
        """ Refer to Parent Class
        """
        weight = self._helper_remove(prefix)
        if len(prefix) == 0:
            self.subtrees = []
            self.weight = 0.0
        else:
            for subtree in self.subtrees:
                old_weight = subtree.weight
                if subtree.value[len(subtree.value) - 1] == prefix[0]:
                    subtree.remove(prefix[1:])
                    if len(subtree.subtrees) == 0:
                        self.subtrees.remove(subtree)
                    if (self.weight_type == 'sum' and subtree.weight !=
                            old_weight) or len(self) == 0:
                        self.weight -= weight[0]
                    elif subtree.weight != old_weight:
                        self.weight = ((self.weight * (len(self) + weight[1])) -
                                       weight[0]) / len(self)

                    self.subtrees.sort(key=lambda x: x.weight, reverse=True)

    def _helper_remove(self, prefix: List) -> List[float, int]:
        """Finds the weight and length of the prefix that is removed
        """
        if len(prefix) == 0:
            if self.weight_type == 'sum':
                return [self.weight, len(self)]
            else:
                return [self.weight * len(self), len(self)]
        else:
            for subtree in self.subtrees:
                if subtree.value[len(subtree.value) - 1] == prefix[0]:
                    return subtree._helper_remove(prefix[1:])
        return []


################################################################################
# CompressedPrefixTree (Task 6)
################################################################################
class CompressedPrefixTree(Autocompleter):
    """A compressed prefix tree implementation.

    While this class has the same public interface as SimplePrefixTree,
    (including the initializer!) this version follows the implementation
    described on Task 6 of the assignment handout, which reduces the number of
    tree objects used to store values in the tree.

    === Attributes ===
    value:
        The value stored at the root of this prefix tree, or [] if this
        prefix tree is empty.
    weight:
        The weight of this prefix tree. If this tree is a leaf, this attribute
        stores the weight of the value stored in the leaf. If this tree is
        not a leaf and non-empty, this attribute stores the aggregate weight
        of the leaf weights in this tree.
    subtrees:
        A list of subtrees of this prefix tree.
    total:
        The total weight of all combined leaves; well be the same as weight if
        weight_type is sum, otherwise it will be different
    weight_type:
        The way of measure for weight being use, either sum of all leaf values
        or the average of all leaf values

    === Representation invariants ===
    - self.weight >= 0

    - (EMPTY TREE):
        If self.weight == 0, then self.value == [] and self.subtrees == [].
        This represents an empty simple prefix tree.
    - (LEAF):
        If self.subtrees == [] and self.weight > 0, this tree is a leaf.
        (self.value is a value that was inserted into this tree.)
    - (NON-EMPTY, NON-LEAF):
        If len(self.subtrees) > 0, then self.value is a list (common prefix),
        and self.weight > 0 (aggregate weight).

    - *NEW*
      This tree does not contain any compressible internal values.
      (See the assignment handout for a definition of "compressible".)

    - self.subtrees does not contain any empty prefix trees.
    - self.subtrees is sorted in non-increasing order of their weights.
      (You can break ties any way you like.)
      Note that this applies to both leaves and non-leaf subtrees:
      both can appear in the same self.subtrees list, and both have a weight
      attribute.
    """
    value: Optional[Any]
    weight: float
    subtrees: List[CompressedPrefixTree]
    total: Optional[float]
    weight_type: str

    def __init__(self, weight_type: str) -> None:
        """Initialize an empty simple prefix tree.

        Precondition: weight_type == 'sum' or weight_type == 'average'.

        The given <weight_type> value specifies how the aggregate weight
        of non-leaf trees should be calculated (see the assignment handout
        for details).
        """
        self.value = []
        self.subtrees = []
        self.weight = 0
        self.weight_type = weight_type
        if self.weight_type == 'average':
            self.total = 0.0

    def is_empty(self) -> bool:
        """Return whether this simple prefix tree is empty."""
        return self.weight == 0.0

    def is_leaf(self) -> bool:
        """Return whether this simple prefix tree is a leaf."""
        return self.weight > 0 and self.subtrees == []

    def __str__(self) -> str:
        """Return a string representation of this tree.

        You may find this method helpful for debugging.
        """
        return self._str_indented()

    def _str_indented(self, depth: int = 0) -> str:
        """Return an indented string representation of this tree.

        The indentation level is specified by the <depth> parameter.
        """
        if self.is_empty():
            return ''
        else:
            s = '  ' * depth + f'{self.value} ({self.weight})\n'
            for subtree in self.subtrees:
                s += subtree._str_indented(depth + 1)
            return s

    def __len__(self) -> int:
        """ Refer to Parent Class
        """
        size = 0
        for subtree in self.subtrees:
            if subtree.is_leaf():
                size += 1
            else:
                size += len(subtree)
        return size

    def insert(self, value: Any, weight: float, prefix: List) -> None:
        """ Refer to Parent Class
        """
        inserted, pref = False, self.value + prefix
        if len(prefix) == 0:
            for subtree in self.subtrees:
                if subtree.value == value:
                    subtree.weight += weight
                    self._update_weight(weight)
                    return
            prefix_tree = CompressedPrefixTree(self.weight_type)
            prefix_tree.value, prefix_tree.weight = value, weight
            self.subtrees.append(prefix_tree)
            self.subtrees.sort(key=lambda x: x.weight, reverse=True)

        else:
            for subtree in self.subtrees:
                num, common = 0, []
                while num < len(subtree.value) and num < (len(pref)):
                    if subtree.value[num] != pref[num]:
                        break
                    else:
                        common.append(pref[num])
                        num += 1
                index = num - len(self.value)
                if subtree.value == common:
                    if len(prefix) == len(self.value) + len(prefix):
                        subtree.insert(value, weight, prefix[num:])
                    else:
                        subtree.insert(value, weight, prefix[index:])
                    inserted = True
                elif common != self.value:
                    new_root = CompressedPrefixTree(self.weight_type)
                    new_root.value = common
                    new_root.insert(value, weight, prefix[index:])
                    new_root.subtrees.append(subtree)
                    if self.weight_type == 'sum':
                        new_root._update_weight(subtree.weight)
                    else:
                        new_root._update_weight(subtree.total)
                    self.subtrees.append(new_root)
                    self.subtrees.remove(subtree)
                    self._update_weight(weight)
                    self.subtrees.sort(key=lambda x: x.weight, reverse=True)
                    return

            if not inserted:
                prefix_tree = CompressedPrefixTree(self.weight_type)
                prefix_tree.value = self.value + prefix
                prefix_tree.insert(value, weight, [])
                self.subtrees.append(prefix_tree)

        self.subtrees.sort(key=lambda x: x.weight, reverse=True)
        self._update_weight(weight)

    def _update_weight(self, weight: float) -> None:
        """ Update the weight
        """
        if self.weight_type == 'sum':
            self.weight += weight
        else:
            self.total += weight
            if len(self) != 0:
                self.weight = self.total / len(self)
            else:
                self.weight = self.total

    def autocomplete(self, prefix: List,
                     limit: Optional[int] = None) -> List[Tuple[Any, float]]:
        """ Refer to Parent Class
        """
        if len(prefix) == 0:
            newlist = []
            for subtree in self.subtrees:
                if limit is None or len(newlist) < limit:
                    if subtree.is_leaf():
                        newlist = newlist + [(subtree.value, subtree.weight)]
                        newlist.sort(key=lambda x: x[1], reverse=True)
                    else:
                        result = subtree.autocomplete([], limit)
                        newlist.extend(result)
                        newlist.sort(key=lambda x: x[1], reverse=True)
            return newlist[:limit]
        else:
            for sub in self.subtrees:
                counter = 0
                common_letters = []
                while counter < len(sub.value) and counter < len(prefix):
                    if sub.value[counter] != prefix[counter]:
                        break
                    else:
                        common_letters.append(prefix[counter])
                        counter += 1
                if common_letters != prefix and common_letters == sub.value:
                    return sub.autocomplete(prefix, limit)
                elif common_letters == sub.value and common_letters == prefix:
                    return sub.autocomplete([], limit)
                elif common_letters == prefix and common_letters != sub.value:
                    return sub.autocomplete([], limit)

    def remove(self, prefix: List) -> None:
        """ Refer to Parent Class
        """
        if len(self.value) > len(prefix):
            return
        elif len(self.value) == len(prefix):
            if self.value != prefix:
                return
            else:
                self.weight = 0
                self.subtrees = []
                return
        else:
            for subtree in self.subtrees:
                num = 0
                inserted = True
                while num < len(subtree.value) and num < len(prefix):
                    if subtree.value[num] != prefix[num]:
                        inserted = False
                        break
                    else:
                        num += 1
                if inserted:
                    if num >= len(prefix):
                        weight = self._helper_remove_compressed(
                            list(subtree.value))
                        subtree.remove(list(subtree.value))
                    else:
                        weight = subtree._helper_remove_compressed(prefix)
                        subtree.remove(prefix)
                    if subtree.is_empty():
                        self.subtrees.remove(subtree)
                    self.update_weight(weight * -1)
        self.subtrees.sort(key=lambda x: x.weight, reverse=True)

    def _helper_remove_compressed(self, prefix: List) -> float:
        """Finds the weight and length of the prefix that is removed
        """
        if len(self.value) > len(prefix):
            if self.weight_type == 'sum':
                return self.weight
            else:
                return self.total
        else:
            for subtree in self.subtrees:
                num1 = 0
                inserted = True
                while num1 < len(subtree.value) and num1 < len(prefix):
                    if subtree.value[num1] != prefix[num1]:
                        inserted = False
                        break
                    else:
                        num1 += 1
                if inserted:
                    return subtree._helper_remove_compressed(prefix[:num1 - 1])
            return 0.0


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'max-nested-blocks': 4
    })
