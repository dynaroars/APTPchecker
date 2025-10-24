import torch.nn.functional as F

from .base import AbstractBase

class AbstractSoftmax(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.options = options.get('softmax', 'complex2')
        self.color = 'cornflowerblue'
        
    def forward(self, x):
        if self.options == 'complex':
            raise NotImplementedError
        return F.softmax(x, dim=self.axis)
        