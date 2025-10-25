from torch import Tensor
import torch._C as _C

class AbstractTensor(Tensor):

    @staticmethod
    def __new__(cls, x, ptb=None, *args, **kwargs):
        if isinstance(x, Tensor):
            tensor = super().__new__(cls, [], *args, **kwargs)
            tensor.data = x.data
            tensor.requires_grad = x.requires_grad
            return tensor
        else:
            return super().__new__(cls, x, *args, **kwargs)
        
    def __init__(self, x, ptb=None):
        self.ptb = ptb

    def __repr__(self):
        if hasattr(self, 'ptb') and self.ptb is not None:
            return f'<AbstractTensor, ptb>'
            return f'<AbstractTensor: {super().__repr__()}, ptb>'
        else:
            return f'<AbstractTensor, no ptb>'
            return f'<AbstractTensor: {super().__repr__()}, no ptb>'
        
    def _func(self, func, *args, **kwargs):
        temp = func(*args, **kwargs)
        new_obj = AbstractTensor([], self.ptb)
        new_obj.data = temp.data
        new_obj.requires_grad = temp.requires_grad
        return new_obj
    
    def to(self, *args, **kwargs):
        if hasattr(self.ptb, 'x_L') and isinstance(self.ptb.x_L, Tensor):
            self.ptb.x_L = self.ptb.x_L.to(*args, **kwargs)
        if hasattr(self.ptb, 'x_U') and isinstance(self.ptb.x_U, Tensor):
            self.ptb.x_U = self.ptb.x_U.to(*args, **kwargs)
        return self._func(super().to, *args, **kwargs)
        
    def clone(self, *args, **kwargs):
        raise NotImplementedError
        
    @classmethod
    def _convert(cls, ret):
        if cls is Tensor:
            return ret

        if isinstance(ret, Tensor):
            return ret
        
        if isinstance(ret, tuple):
            ret = tuple(cls._convert(r) for r in ret)

        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        assert all(issubclass(cls, t) for t in types)
        with _C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            return cls._convert(ret)
