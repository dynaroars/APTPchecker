from .base import AbstractBase

class AbstractExpand(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
    def forward(self, x, y):
        y = y.clone()
        assert y.ndim == 1
        n, m = x.ndim, y.shape[0]
        assert n <= m
        for i in range(n):
            if y[m - n + i] == 1:
                y[m - n + i] = x.shape[i]
            else:
                assert x.shape[i] == 1 or x.shape[i] == y[m - n + i]
        return x.expand(*list(y))
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars