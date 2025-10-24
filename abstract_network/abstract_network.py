from collections import OrderedDict
from beartype import beartype
from torch import nn
import typing
import torch
import copy

from .helper import unpack_inputs, parse_module, Node
from .constant import abstract_op_mapping
from .bound import BoundedTensor
from .operator import *

class AbstractNetwork(nn.Module):

    @beartype
    def __init__(self, model, input_shape: tuple, device: str):
        super().__init__()
        self.input_shape = input_shape
        self.device = device

        model.eval()
        model.to(self.device)
        self.bound_opts = {}
        
        object.__setattr__(self, 'ori_state_dict', copy.deepcopy(model.state_dict()))
        
        default_input = torch.zeros(input_shape, device=device)
        # default_output = model(default_input)
        self._convert(model, default_input)
        
    # DONE
    def _to(self, obj, dest, inplace=False):
        """ Move all tensors in the object to a specified dest (device or dtype). The inplace=True option is available for dict."""
        if obj is None:
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.to(dest)
        elif isinstance(obj, tuple):
            return tuple([self._to(item, dest) for item in obj])
        elif isinstance(obj, list):
            return list([self._to(item, dest) for item in obj])
        elif isinstance(obj, dict):
            if inplace:
                for k, v in obj.items():
                    obj[k] = self._to(v, dest, inplace=True)
                return obj
            else:
                return {k: self._to(v, dest) for k, v in obj.items()}
        else:
            raise NotImplementedError(type(obj))
        
    @beartype
    def _convert_nodes(self, model, global_input: torch.Tensor) -> tuple[list[AbstractBase], list[AbstractBase], list[AbstractBase], typing.Any]:
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
                # elif nodesOP[n].op.startswith('aten::ATen'):
                    # op = eval(f'AbstractATen{attr["operator"].capitalize()}')
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
                    inputs=self._get_node_input(nodesOP, nodesIn, nodesOP[n]), 
                    output_index=nodesOP[n].output_index, 
                    options=self.bound_opts,
                )
            )
            
            # visualize only
            nodesOP[n].bound_node.ori_name = nodesOP[n].op
            
        # assign node name
        for node in nodesIn + nodesOP:
            node.bound_node.name = node.name
            # print(node.bound_node.name)

        nodes_dict = {}
        for node in nodesOP + nodesIn:
            nodes_dict[node.name] = node.bound_node
            
        nodesOP = [n.bound_node for n in nodesOP]
        nodesIn = [n.bound_node for n in nodesIn]
        nodesOut = [nodes_dict[n] for n in nodesOut]
        return nodesOP, nodesIn, nodesOut, template
        
        
    # DONE
    def _get_node_name_map(self):
        """Build a dict with {ori_name: name, name: ori_name}"""
        self.node_name_map = {}
        for node in self.nodes():
            if isinstance(node, (AbstractInput, AbstractParameter)):
                for p in list(node.named_parameters()):
                    if node.ori_name not in self.node_name_map:
                        name = f'{node.name}.{p[0]}'
                        self.node_name_map[node.ori_name] = name
                        self.node_name_map[name] = node.ori_name
                for p in list(node.named_buffers()):
                    if node.ori_name not in self.node_name_map:
                        name = f'{node.name}.{p[0]}'
                        self.node_name_map[node.ori_name] = name
                        self.node_name_map[name] = node.ori_name
                        
        # print(self.node_name_map)


    @beartype
    def _convert(self, model, global_input: torch.Tensor):
        nodesOP, nodesIn, nodesOut, template = self._convert_nodes(model, global_input)
        global_input = self._to(global_input, self.device)
        global_input = (global_input,)
        while True:
            self.build_graph(nodesOP, nodesIn, nodesOut, template)
            self.forward(*global_input)  # running means/vars changed
            
            nodesOP, nodesIn, finished = self._split_complex(nodesOP, nodesIn)
            if finished:
                break
            
        self._load_state_dict()
    
    # DONE
    def _load_state_dict(self):
        if hasattr(self, 'ori_state_dict'):
            self._get_node_name_map()
            ori_state_dict_mapped = OrderedDict()
            for k, v in self.ori_state_dict.items():
                if k in self.node_name_map:
                    ori_state_dict_mapped[self.node_name_map[k]] = v
                    # print(f'Load: {k} {self.node_name_map[k]}')
            self.load_state_dict(ori_state_dict_mapped)
    
    def _clear_attr(self):
        for l in self.nodes():
            for attr in ['forward_value']:
                if hasattr(l, attr):
                    delattr(l, attr)
                    
    # DONE
    def set_input(self, *x):
        self._clear_attr()
        inputs_unpacked = unpack_inputs(x)
        # print('[+] Set new input', type(inputs_unpacked[0]), self.input_name, self.input_index)
        for name, index in zip(self.input_name, self.input_index):
            if index is None:
                continue
            node = self[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, BoundedTensor):
                node.perturbation = node.value.ptb
                # print('\t- set:', node, node.perturbation)
            else:
                node.perturbation = None
    
    @beartype
    def get_forward_value(self, node: AbstractBase):
        if getattr(node, 'forward_value', None) is not None:
            # print(f'\t - Skip: {node}')
            return node.forward_value

        inputs = [self.get_forward_value(inp) for inp in node.inputs]

        node.input_shape = inputs[0].shape if len(inputs) > 0 else None
        fv = node.forward(*inputs)
        if isinstance(fv, (torch.Size, tuple)):
            fv = torch.tensor(fv, device=self.device)
        node.forward_value = fv
        # print('set foward value', node.forward_value)
        node.output_shape = fv.shape
        # print(f'[+] Forward:  {node}')# {node.input_shape=} {node.output_shape=}')
        return fv
    
    # DONE
    def forward(self, *x, final_node_name=None):
        self.set_input(*x)
        if final_node_name is None:
            assert len(self.output_name) == 1
            final_node_name = self.output_name[0]
        return self.get_forward_value(self[final_node_name])
        
    def _split_complex(self, nodesOP, nodesIn):
        finished = True
        return nodesOP, nodesIn, finished
        
    # DONE
    @beartype
    def _get_node_input(self, nodesOP: list[Node], nodesIn: list[Node], node: Node) -> list[AbstractBase]:
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
    
    # DONE
    def nodes(self):
        return self._modules.values()
    
    def roots(self):
        return [self[name] for name in self.root_names]

    # DONE
    def __getitem__(self, name):
        module = self._modules[name]
        # We never create modules that are None, the assert fixes type hints
        assert module is not None
        return module
    
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
        
    def final_node(self):
        return self[self.final_name]
    
    @beartype
    def get_hidden_bounds(self, names: list[str]) -> tuple[dict, dict]:
        lower_bounds, upper_bounds = {}, {}
        for name in names:
            assert self[name].lower is not None
            assert self[name].upper is not None
            lower_bounds[name] = self[name].lower
            upper_bounds[name] = self[name].upper
        return lower_bounds, upper_bounds
    
    
    from .helper import build_graph, visualize
    from .solver.mip_solver import build_solver_module 

    