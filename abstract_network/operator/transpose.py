import numpy as np
import torch

from .base import AbstractBase

class AbstractTranspose(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.perm = attr['perm']
        
    def forward(self, x):
        return x.permute(*self.perm)


    def build_solver(self, *v, model, C=None):
        if isinstance(v[0], torch.Tensor):
            self.solver_vars = self.forward(*v)
        else:
            assert len(v[0]) == 1, f'{len(v[0])=}'
            self.solver_vars = np.array(v[0]).transpose(self.perm)
        return self.solver_vars