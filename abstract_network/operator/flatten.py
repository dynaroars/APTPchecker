import numpy as np
import torch

from .base import AbstractBase

class AbstractFlatten(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return torch.flatten(x, self.attr['axis'])
    
    def build_solver(self, *v, model, C=None):
        assert self.attr['axis'] == 1
        assert len(v[0]) == 1, f'{len(v[0])=}'
        self.solver_vars = np.array(v[0]).reshape(1, -1)
        return self.solver_vars
    