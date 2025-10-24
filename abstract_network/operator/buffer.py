from .input import AbstractInput

class AbstractBuffer(AbstractInput):
    
    '''Constant Initializers: (nn.Parameter, BatchNorm, etc.)'''
    
    def __init__(self, ori_name, value, perturbation=None, options=None, attr=None):
        super().__init__(ori_name, None, perturbation, attr=attr)
        # AbstractBuffer are like constants and they are by default not from inputs.
        self.register_buffer('buffer', value.clone().detach())
        # print(f'\t- [+] Init buffer: {ori_name=} {self.buffer=}')
        
        # visualize only
        self.color = 'aquamarine'

    def forward(self):
        return self.buffer.to(self.device)
