__version__ = "0.0.1"
__author__ = 'Tullio Pizzuti'
__credits__ = 'DAIS Lab (Computer Science Department, University of Salerno)'

from typing import List, Iterable, Union
from pyroaring import BitMap

class HashableBitMap(BitMap):

    def __str__(self):
        return f'{str(list(self.to_array()))}'

    def bitset_to_tuple(self):
        return tuple(self.to_array())

    def __hash__(self):
        return hash(self.bitset_to_tuple())

    def __eq__(self, other):
        if not isinstance(other, HashableBitMap):
            return NotImplemented
        return self.bitset_to_tuple() == other.bitset_to_tuple()

    def bitset_to_int(self):
        int_value = 0
        for i in self:
            int_value |= 1 << i
        return int_value

    def __lt__(self, other):
        if not isinstance(other, HashableBitMap):
            return NotImplemented
        return self.bitset_to_int() < other.bitset_to_int()

    def __le__(self, other):
        if not isinstance(other, HashableBitMap):
            return NotImplemented
        return self.bitset_to_int() <= other.bitset_to_int()

    def __gt__(self, other):
        if not isinstance(other, HashableBitMap):
            return NotImplemented
        return self.bitset_to_int() > other.bitset_to_int()

    def __ge__(self, other):
        if not isinstance(other, HashableBitMap):
            return NotImplemented
        return self.bitset_to_int() >= other.bitset_to_int()
