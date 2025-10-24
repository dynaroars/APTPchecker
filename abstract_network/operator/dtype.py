import torch

from .base import AbstractBase

class AbstractCast(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
        # See values of enum DataType in TensorProto.
        # Unsupported: str, uint16, uint32, uint64.
        data_types = [
            None,  torch.float, torch.uint8, torch.int8,
            None,  torch.int16, torch.int32, torch.int64,
            None,  torch.bool, torch.float16, torch.float64,
            None,  None, torch.complex64, torch.complex128,
        ]
        self.type = data_types[attr['to']]
        assert self.type is not None, "Unsupported type conversion."
        
    def forward(self, x):
        return x.to(self.type)


    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars