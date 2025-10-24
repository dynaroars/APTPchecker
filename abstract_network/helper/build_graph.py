from __future__ import annotations
from typing import TYPE_CHECKING
from beartype import beartype

from ..operator import AbstractBase

if TYPE_CHECKING:
    import abstract_network

    
# Make sure the nodes already have `name` and `input_name`
@beartype
def _add_nodes(self: abstract_network.AbstractNetwork, nodes: list[AbstractBase]) -> None:
    nodes = [(node if isinstance(node, AbstractBase) else node.bound_node) for node in nodes]
    for node in nodes:
        # check duplicate names
        if node.name in self._modules:
            raise NameError(f'Node with name {node.name} already exists')
        self._modules[node.name] = node
        if len(node.inputs) == 0:
            self.root_names.append(node.name)
    for node in nodes:
        node.output_name = []
        for l_pre in node.inputs:
            l_pre.output_name.append(node.name)


@beartype
def _add_input_node(self: abstract_network.AbstractNetwork, node: AbstractBase, index: int | None = None) -> None:
    _add_nodes(self, [node])
    self.input_name.append(node.name)
    # default value for input_index
    if index == 'auto':
        index = max([0] + [(i + 1) for i in self.input_index if i is not None])
    self.input_index.append(index)


@beartype
def build_graph(self: abstract_network.AbstractNetwork, nodesOP: list[AbstractBase], nodesIn: list[AbstractBase], nodesOut: list[AbstractBase], template: None) -> None:
    # We were assuming that the original model had only one output node.
    assert len(nodesOut) == 1
    self.final_name = nodesOut[0].name
    self.input_name, self.input_index, self.root_names = [], [], []
    self.output_name = [n.name for n in nodesOut]
    self.output_template = template
    self._modules.clear()
    for node in nodesIn:
        _add_input_node(self, node, index=node.input_index)
    _add_nodes(self, nodesOP)
    self.root_names = [node.name for node in nodesIn]
