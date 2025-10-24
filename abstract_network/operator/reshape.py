from math import prod
import numpy as np
import torch

from .base import AbstractBase

class AbstractReshape(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # It can be set to `view`, so that `view` instead of `reshape` will be used.
        self.options = options.get('reshape', 'reshape')

    def forward(self, x, shape):
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = prod(x.shape) // int(prod(shape[:i]) * prod(shape[(i + 1):]))
        
        if self.options == 'view':
            return x.contiguous().view(shape)
        return x.reshape(shape)
    
    
    def build_solver(self, *v, model, C=None):
        assert isinstance(v[1], torch.Tensor)
        if isinstance(v[0], torch.Tensor):
            self.solver_vars = self.forward(*v)
        else:
            assert len(v[0]) == 1, f'{len(v[0])=}'
            gvar_array = np.array(v[0])
            gvar_array = gvar_array.reshape(v[1].detach().cpu().numpy())
            self.solver_vars = gvar_array
        return self.solver_vars