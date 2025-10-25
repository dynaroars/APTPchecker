from .input import AbstractInput

class AbstractBuffer(AbstractInput):
    
    '''Constant Initializers: (nn.Parameter, BatchNorm, etc.)'''
    
    def __init__(self, ori_name, value, perturbation=None, options=None, attr=None):
        super().__init__(ori_name, None, perturbation, attr=attr)
        self.register_buffer('buffer', value.clone().detach())
        self.color = 'aquamarine' # visualize only

    def forward(self):
        return self.buffer.to(self.device)
