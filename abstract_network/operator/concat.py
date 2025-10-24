import gurobipy as grb
import numpy as np
import torch

from .base import AbstractBase

class AbstractConcat(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        
    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, torch.Tensor) else torch.tensor(item)) for item in x]
        self.axis = self.make_axis_non_negative(self.axis)
        return torch.cat(x, dim=int(self.axis))
    
    
    def build_solver(self, *v, model, C=None):
        if any(isinstance(_, grb.Var) for v_ in v for _ in np.array(v_).reshape(-1)):
            gvar_array = [np.array(_) for _ in v]
            self.solver_vars = np.concatenate(gvar_array, axis=self.axis)
        else:
            self.solver_vars = self.forward(*v)
        return self.solver_vars
