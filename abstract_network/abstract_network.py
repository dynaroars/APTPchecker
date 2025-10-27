from beartype import beartype
from torch import nn
import torch
import copy

from .helper import unpack_inputs
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
        self.convert(model, default_input)
        
    @beartype
    def convert(self, model, global_input: torch.Tensor):
        nodesOP, nodesIn, nodesOut, template = self.convert_nodes(model, global_input)
        global_input = (self._to(global_input, self.device),)
        self.build_graph(nodesOP, nodesIn, nodesOut, template)
        self.forward(*global_input)  # running means/vars changed
        # self._load_state_dict()
        
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
        
    # # DONE
    # def _get_node_name_map(self):
    #     """Build a dict with {ori_name: name, name: ori_name}"""
    #     self.node_name_map = {}
    #     for node in self.nodes():
    #         if isinstance(node, (AbstractInput, AbstractParameter)):
    #             for p in list(node.named_parameters()):
    #                 if node.ori_name not in self.node_name_map:
    #                     name = f'{node.name}.{p[0]}'
    #                     self.node_name_map[node.ori_name] = name
    #                     self.node_name_map[name] = node.ori_name
    #             for p in list(node.named_buffers()):
    #                 if node.ori_name not in self.node_name_map:
    #                     name = f'{node.name}.{p[0]}'
    #                     self.node_name_map[node.ori_name] = name
    #                     self.node_name_map[name] = node.ori_name
                        
    #     # print(self.node_name_map)
    
    # # DONE
    # def _load_state_dict(self):
    #     if hasattr(self, 'ori_state_dict'):
    #         self._get_node_name_map()
    #         ori_state_dict_mapped = OrderedDict()
    #         for k, v in self.ori_state_dict.items():
    #             if k in self.node_name_map:
    #                 ori_state_dict_mapped[self.node_name_map[k]] = v
    #                 # print(f'Load: {k} {self.node_name_map[k]}')
    #         self.load_state_dict(ori_state_dict_mapped)
    
    def _clear_attr(self):
        for l in self.nodes():
            for attr in ['forward_value']:
                if hasattr(l, attr):
                    delattr(l, attr)
                    
    # DONE
    def set_input(self, *x):
        self._clear_attr()
        inputs_unpacked = unpack_inputs(x)
        for name, index in zip(self.input_name, self.input_index):
            if index is None:
                continue
            node = self[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, AbstractTensor):
                node.perturbation = node.value.ptb
            else:
                node.perturbation = None
    
    @beartype
    def get_forward_value(self, node: AbstractBase):
        if getattr(node, 'forward_value', None) is not None:
            return node.forward_value
        inputs = [self.get_forward_value(inp) for inp in node.inputs]
        node.input_shape = inputs[0].shape if len(inputs) > 0 else None
        fv = node.forward(*inputs)
        if isinstance(fv, (torch.Size, tuple)):
            fv = torch.tensor(fv, device=self.device)
        node.forward_value = fv
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
        
        
    # DONE
    def nodes(self):
        return self._modules.values()
    
    def roots(self):
        return [self[name] for name in self.root_names]

    # DONE
    def __getitem__(self, name):
        module = self._modules[name]
        assert module is not None
        return module
    
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
        
    def final_node(self):
        return self[self.final_name]
    
    def split_nodes(self):
        """Activation functions that can be split during branch and bound."""
        return [n for n in self.nodes() if n.splittable]
    
    from .helper import build_graph, visualize, convert_nodes
    from .solver.mip_solver import build_solver_module 

    