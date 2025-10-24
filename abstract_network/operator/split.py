import torch

from .base import AbstractBase

class AbstractSplit(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.split = attr.get('split', None)
        
    def forward(self, *x):
        data = x[0]
        split = self.split if self.split is not None else x[1].tolist()
        if self.axis == -1:
            self.axis = len(data.shape) - 1
        return torch.split(data, split, dim=self.axis)[self.output_index]
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars