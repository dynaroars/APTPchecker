from __future__ import annotations
from typing import TYPE_CHECKING
from beartype import beartype
import torch.nn as nn
import typing
import torch

from .op_mapping import abstract_op_mapping
from .parser import parse_module, Node
from ..operator import *


if TYPE_CHECKING:
    import abstract_network
    
# DONE
@beartype
def _get_node_inputs(nodesOP: list[Node], nodesIn: list[Node], node: Node) -> list[AbstractBase]:
    ret = []
    for i in range(len(node.inputs)):
        for op in nodesOP:
            if op.name == node.inputs[i]:
                ret.append(op.bound_node)
                break
        if len(ret) == i + 1:
            continue
        for io in nodesIn:
            if io.name == node.inputs[i]:
                ret.append(io.bound_node)
                break
        if len(ret) <= i:
            raise ValueError(f'cannot find inputs of node: {node.name}')
    return ret

@beartype
def convert_nodes(self: abstract_network.AbstractNetwork, model, global_input: torch.Tensor) -> tuple[list[AbstractBase], list[AbstractBase], list[AbstractBase], typing.Any]:
    r"""
    Returns:
        nodesOP (list): List of operator nodes
        nodesIn (list): List of input nodes
        nodesOut (list): List of output nodes
        template (object): Template to specify the output format
    """
    global_input_cpu = self._to(global_input, 'cpu')
    model.to('cpu')
    nodesOP, nodesIn, nodesOut, template = parse_module(model, global_input_cpu)
    model.to(self.device)
    
    for i in range(0, len(nodesIn)):
        if nodesIn[i].param is not None:
            nodesIn[i] = nodesIn[i]._replace(param=nodesIn[i].param.to(self.device))
    
    # convert input nodes
    attr = {'device': self.device}
    for i, n in enumerate(nodesIn):
        if n.input_index is not None:
            assert i == 0
            assert n.input_index == 0
            nodesIn[i] = nodesIn[i]._replace(
                bound_node=AbstractInput(
                    ori_name=n.ori_name,
                    value=global_input,
                    perturbation=n.perturbation,
                    input_index=n.input_index, 
                    options=self.bound_opts,
                    attr=attr,
                )
            )
        else:
            assert isinstance(nodesIn[i].param, (nn.Parameter, torch.Tensor)), f'{type(nodesIn[i].param)=} {nodesIn[i].param=} {nodesIn[i].ori_name=}'
            bound_class = AbstractParameter if isinstance(nodesIn[i].param, nn.Parameter) else AbstractBuffer
            nodesIn[i] = nodesIn[i]._replace(
                bound_node=bound_class(
                    ori_name=n.ori_name, 
                    value=n.param,
                    perturbation=n.perturbation, 
                    options=self.bound_opts,
                    attr=attr,
                )
            )
        # print(i, str(nodesIn[i].bound_node))
            
    # convert op nodes.
    for n in range(len(nodesOP)):
        attr = nodesOP[n].attr
        try:
            if nodesOP[n].op in abstract_op_mapping:
                op = abstract_op_mapping[nodesOP[n].op]
            elif nodesOP[n].op.startswith('onnx::'):
                op = eval(f'Abstract{nodesOP[n].op[6:]}')
            else:
                raise NotImplementedError()
        except NameError:
            print(f'[!] The node has an unsupported operation: {nodesOP[n]}')
            exit()
        except:
            raise
            
        # print(f'[+] Creating: {nodesOP[n].op=} {op=} {nodesOP[n].name}')
        attr['device'] = self.device
        nodesOP[n] = nodesOP[n]._replace(
            bound_node=op(
                attr=attr, 
                inputs=_get_node_inputs(nodesOP, nodesIn, nodesOP[n]), 
                output_index=nodesOP[n].output_index, 
                options=self.bound_opts,
            )
        )
        
        nodesOP[n].bound_node.ori_name = nodesOP[n].op # visualize only
        
    # assign node name
    for node in nodesIn + nodesOP:
        node.bound_node.name = node.name

    nodes_dict = {}
    for node in nodesOP + nodesIn:
        nodes_dict[node.name] = node.bound_node
        
    nodesOP = [n.bound_node for n in nodesOP]
    nodesIn = [n.bound_node for n in nodesIn]
    nodesOut = [nodes_dict[n] for n in nodesOut]
    return nodesOP, nodesIn, nodesOut, template
    