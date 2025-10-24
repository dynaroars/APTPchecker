from beartype import beartype
import torch.nn as nn
import torch

def not_implemented_op(node, func):
    message = f'Function `{func}` of `{node}` is not supported yet.'
    raise NotImplementedError(message)


class AbstractBase(nn.Module):


    @beartype
    def __init__(self, attr: None | dict = None, inputs: None | list = None, output_index: int = 0, options: None | dict = None):
        super().__init__()
        self.attr = attr if attr is not None else {}
        self.options = options if options is not None else {}
        self.inputs = inputs if inputs is not None else []
        self.output_index = output_index
        self.output_name = []
        self.device = attr.get('device')
        
        # node 
        self.name = None # node name for retrieval/debug
        self.ori_name = None # node name for retrieval/debug
        self.input_shape = None
        self.output_shape = None
        self.perturbation = None
        self.lower = None
        self.upper = None
        
        # input
        self.input_index = None # list of input indices, e.g., [0, None, ...]
        self.value = None
        
        # visualize only
        self.color = "white"
        
        # capture attributes set during initialization
        self._allowed_attributes = {
            'forward_value', # forward value of each layer
            'transA', 'transB', 'alpha_linear', 'beta_linear', # Linear
            'F_conv', 'stride', 'padding', 'dilation', 'groups', 'has_bias', # Conv
            'leaky_alpha', # ReLU
            'axis', # Gather, Concat, Split, Softmax
            'axes', # Squeeze, Unsqueeze, Reduce
            'keepdim', # Reduce
            'split', # Split
            'perm', # Transpose
            'epsilon', 'momentum', 'use_mean', 'use_var', 'use_affine', # BatchNorm
            'type', # Cast
            'kernel_size', 'ceil_mode', 'count_include_pad', # Pooling
            'solver_vars', # mip solver
        }
        self._initialized = True


    
    def make_axis_non_negative(self, axis, shape='input'):
        if isinstance(axis, (tuple, list)):
            return tuple([self.make_axis_non_negative(item, shape) for item in axis])
        if shape == 'input':
            assert self.input_shape is not None
            shape = self.input_shape
        elif shape == 'output':
            assert self.output_shape is not None
            shape = self.output_shape
        else:
            assert isinstance(shape, torch.Size)
        if axis < 0:
            return axis + len(shape)
        else:
            return axis
        
    def __setattr__(self, key, value):
        if not hasattr(self, '_initialized') or hasattr(self, key) or key in self._allowed_attributes:
            # Allow setting attributes during initialization or if the attribute already exists
            super().__setattr__(key, value)
            # print(f'[!] Set {key=} {value=}')
        else:
            raise ValueError(f"`{key}` is not an allowed attribute of `{self}`")
        
        
    def __repr__(self, attrs=None):
        inputs = ', '.join([str(node.name) for node in self.inputs])
        ret = f'{self.__class__.__name__}(name={self.name}, inputs=[{inputs}]'
        if attrs is not None:
            for k, v in attrs.items():
                ret += f', {k}={v}'
        ret += ')'
        return ret