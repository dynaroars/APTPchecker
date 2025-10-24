from torch import Tensor
import gurobipy as grb
import numpy as np
import torch

from .base import AbstractBase

class AbstractLinear(AbstractBase):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        ''' 
            A = A if transA == 0 else A.T
            B = B if transB == 0 else B.T
            C = C if C is not None else np.array(0)
            output = alpha * np.dot(A, B) + beta * C
        '''

        super().__init__(attr, inputs, output_index, options)

        # Defaults in ONNX
        self.transA = 0
        self.transB = 0
        self.alpha_linear = 1.0
        self.beta_linear = 1.0
        if attr is not None:
            self.transA = attr['transA'] if 'transA' in attr else self.transA
            self.transB = attr['transB'] if 'transB' in attr else self.transB
            self.alpha_linear = attr['alpha'] if 'alpha' in attr else self.alpha_linear
            self.beta_linear = attr['beta'] if 'beta' in attr else self.beta_linear


    def _preprocess(self, a, b, c=None):
        """Handle tranpose and linear coefficients."""
        if self.transA and isinstance(a, Tensor):
            a = a.transpose(-2, -1)
        if self.alpha_linear != 1.0:
            a = self.alpha_linear * a
        if not self.transB and isinstance(b, Tensor):
            # our code assumes B is transposed (common case), 
            # so we transpose B only when it is not transposed in gemm.
            b = b.transpose(-2, -1)
        if c is not None:
            if self.beta_linear != 1.0:
                c = self.beta_linear * c
        return a, b, c
    
    def forward(self, x, w, b=None):
        x, w, b = self._preprocess(x, w, b)
        # self.input_shape = self.x_shape = x.shape
        # self.y_shape = w.t().shape
        res = x.matmul(w.t())
        if b is not None:
            res += b
        return res
    

    def build_solver(self, *v, model, C=None):
        has_bias = self is not None and len(v) == 3
        
        # pre-check
        assert len(v[0]) == 1, f'{len(v[0])=}'
        gvars_array = np.array(v[0]) 
        # TODO: generalize
        assert len(gvars_array.shape) == 2, f'{gvars_array.shape=}'
        assert not any((_.lb == -float('inf') or _.ub == float('inf')) for _ in gvars_array.reshape(-1))

        # current layer weight
        this_layer_weight = v[1]
        if self.transB == 0:
            this_layer_weight = this_layer_weight.transpose(1, 0)
        if C is not None:
            this_layer_weight = C.squeeze(0).mm(this_layer_weight)
        this_layer_weight = this_layer_weight.detach().cpu().numpy()

        # current layer bias
        this_layer_bias = None
        if has_bias:
            this_layer_bias = v[2]
            if C is not None:
                this_layer_bias = C.squeeze(0).mm(this_layer_bias.unsqueeze(-1)).view(-1)
            this_layer_bias = this_layer_bias.detach().cpu().numpy()

        # current layer constraints
        new_layer_gurobi_vars = []
        for neuron_idx in range(len(this_layer_weight)):
            lin_expr = this_layer_bias[neuron_idx].item() if has_bias else 0
            coeffs = this_layer_weight[neuron_idx]
            lin_expr += grb.LinExpr(coeffs, v[0][0])
            
            var = model.addVar(
                lb=-float('inf'), 
                ub=float('inf'), 
                obj=0, 
                vtype=grb.GRB.CONTINUOUS, 
                name=f'lay{self.name}_{neuron_idx}',
            )
            model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        self.solver_vars = np.array([new_layer_gurobi_vars])
        model.update()
        return self.solver_vars
    
    
    
    
class AbstractMatMul(AbstractLinear):
    
    # Reuse most functions from BoundLinear.
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.transA = 0
        self.transB = 0
        
    def forward(self, x, y):
        return x.matmul(y)
    
    def build_solver(self, *v, model, C=None):
        
        if len(v[0].shape) == 2:
            return super().build_solver(*v, model=model, C=C)
        
        # pre-check
        assert len(v[0]) == 1
        assert len(v[0]) == 1, f'{len(v[0])=}'
        assert C is None
        gvars_array = np.array(v[0]) 
        assert not any((_.lb == -float('inf') or _.ub == float('inf')) for _ in gvars_array.reshape(-1))
    
        # current layer weight
        this_layer_weight = v[1]
        if isinstance(this_layer_weight, torch.Tensor):
            this_layer_weight = this_layer_weight.detach().cpu().numpy()

        # current layer constraints
        neuron_idx = 0
        new_layer_gurobi_vars = []
        this_layer_shape = self.output_shape
        if len(v[0].shape) == 3:
            assert len(this_layer_weight.shape) == 2 # TODO: generalize
            # (1, 5, 4) x (4, 6) = (1, 5, 6) 
            for chan1 in range(this_layer_shape[1]):
                out_chan1_vars = []
                for chan2 in range(this_layer_shape[2]):
                    coeffs = this_layer_weight[:, chan2]
                    lin_expr = grb.LinExpr(coeffs, v[0][0, chan1])

                    var = model.addVar(
                        lb=-float('inf'), 
                        ub=float('inf'), 
                        obj=0, 
                        vtype=grb.GRB.CONTINUOUS, 
                        name=f'lay{self.name}_{neuron_idx}',
                    )
                    model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1
                    out_chan1_vars.append(var)
                new_layer_gurobi_vars.append(out_chan1_vars)
            
        elif len(v[0].shape) == 4:
            # attention QKV
            assert len(v[1].shape) == 4
            # (1, 5, 4, 3) x (1, 5, 3, 6) = (1, 5, 4, 6) 
            for chan1 in range(this_layer_shape[1]):
                out_chan1_vars = []
                for chan2 in range(this_layer_shape[2]):
                    out_chan2_vars = []
                    for chan3 in range(this_layer_shape[3]):
                        gvars1 = v[0][0, chan1, chan2, :]
                        gvars2 = this_layer_weight[0, chan1, :, chan3]
                        assert len(gvars1) == len(gvars2)
                        expr = 0.0
                        for cidx in range(len(gvars1)):
                            expr += gvars1[cidx] * gvars2[cidx]
                        
                        var = model.addVar(
                            lb=-float('inf'), 
                            ub=float('inf'), 
                            obj=0, 
                            vtype=grb.GRB.CONTINUOUS, 
                            name=f'lay{self.name}_{neuron_idx}',
                        )
                        model.addConstr(expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                        neuron_idx += 1
                        out_chan2_vars.append(var)
                    out_chan1_vars.append(out_chan2_vars)
                new_layer_gurobi_vars.append(out_chan1_vars)
        else:
            raise NotImplementedError(f'{v[0].shape=}')
        
        assert np.array(new_layer_gurobi_vars).shape == this_layer_shape[1:]
        self.solver_vars = np.array([new_layer_gurobi_vars])
        model.update()
        return self.solver_vars
    
    def _is_softmax(self):
        """This multiplication comes from softmax."""
        raise NotImplementedError
        
    