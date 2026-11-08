__version__ = "0.0.1"
__author__ = 'Tullio Pizzuti'
__credits__ = 'DAIS Lab (Computer Science Department, University of Salerno)'

from abc import abstractmethod
from typing import Generic, TypeVar, Iterable, List, FrozenSet

from settrie import SetTrie

from hashablebitmap import HashableBitMap

T=TypeVar('T')
N=TypeVar('N')
class OperationResult(Generic[T]):
    def __init__(self,result:T|None=None, nodes_to_add:Iterable[N]|None=None, nodes_to_remove:Iterable[FrozenSet[int]]|None=None):
        self.result = result
        self.nodes_to_add = nodes_to_add
        self.nodes_to_remove = nodes_to_remove


class Node(HashableBitMap):
    max_dimension: int
    excluded_nodes: HashableBitMap

    def __new__(cls, comb: Iterable[int], max_dimension: int, excluded_nodes: HashableBitMap | None = None):
        return super(Node, cls).__new__(cls, comb)

    def __init__(self, comb: Iterable[int], max_dimension: int, excluded_nodes: HashableBitMap | None = None):
        super().__init__()
        self.comb = comb
        self.max_dimension = max_dimension
        self.excluded_nodes = HashableBitMap(comb) if (
                    excluded_nodes is None or len(excluded_nodes) <= 0) else excluded_nodes|self



    def __str__(self):
        return f"({super(Node, self).__str__()}:{str(self.excluded_nodes)})"

    def __repr__(self):
        return str(self)

    @abstractmethod
    def operation(self) -> T:
        ...

    @abstractmethod
    def update_combination_to_exclude(self, result: T):
        ...

    @abstractmethod
    def cast_node(self, node: HashableBitMap) -> "Node":
        ...

    def cast_nodes(self, nodes: Iterable) -> Iterable["Node"]:
        return map(lambda to_cast: self.cast_node(to_cast), nodes)

    def generate_next_combinations(self, result: T) -> OperationResult[T]:
        final_result = OperationResult[T]()
        final_result.result = result
        if len(self.excluded_nodes) == self.max_dimension:
            final_result.nodes_to_add = None
            final_result.nodes_to_remove = [self]
            return final_result

        final_result.nodes_to_remove = list(
            map(lambda to_remove: to_remove | self, map(lambda e: HashableBitMap([e]), self.excluded_nodes ^ self)))
        final_result.nodes_to_add = list(self.cast_nodes(map(lambda to_add: to_add | self,
                                                             map(lambda e: HashableBitMap([e]),
                                                                 self.excluded_nodes.flip(0, self.max_dimension)))))
        for edit_node in final_result.nodes_to_add:
            edit_node.excluded_nodes=edit_node.excluded_nodes | self.excluded_nodes

        return final_result

    def perform_operation(self) -> OperationResult[T]:
        result = self.operation()
        self.update_combination_to_exclude(result)
        return self.generate_next_combinations(result)


class LatticeLevel(dict[HashableBitMap, Node]):

    def __init__(self, level: int, results_saver: List|None, excluded_nodes: SetTrie):
        super().__init__()
        self.level = level
        self.results_saver = results_saver
        self.excluded_nodes = excluded_nodes

    def add_excluded_node(self, node: HashableBitMap):
        if not self.excluded_nodes.hassubset(node):
            self.excluded_nodes.add(node)

    def add_excluded_nodes(self, nodes: Iterable[HashableBitMap]):
        if nodes:
            [self.add_excluded_node(to_exclude) for to_exclude in nodes]

    def add_node(self, node: Node):
        if node not in self.excluded_nodes:
            if node in self:
                self[HashableBitMap(node)].excluded_nodes |= node.excluded_nodes
            else:
                self[HashableBitMap(node)] = node

    def add_nodes(self, nodes: Iterable[Node]):
        if nodes:
            [self.add_node(to_add) for to_add in nodes]

    def perform_operation(self) -> "LatticeLevel":
        new_level = LatticeLevel(self.level + 1, self.results_saver, self.excluded_nodes)
        results = filter(lambda node_result: node_result is not None, map(lambda node: node.perform_operation(), filter(
            lambda node: not self.excluded_nodes.hassubset(node), self.values())))
        for result in results:
            if new_level.results_saver is not None and result.result is not None:
                new_level.results_saver.append(result.result)
            new_level.add_excluded_nodes(result.nodes_to_remove)
            new_level.add_nodes(result.nodes_to_add)
        return new_level

    def is_not_empty(self):
        return len(self) > 0


class NoopNode(Node):

    def operation(self) -> T:
        return None

    def update_combination_to_exclude(self, result: T):
        pass

    def cast_node(self, node: HashableBitMap) -> "NoopNode":
        return NoopNode(node, self.max_dimension)