from beartype import beartype
import multiprocessing
import gurobipy as grb
import numpy as np
import torch
import time
import sys
import os

from ..operator import AbstractBase, AbstractInput, AbstractLinear, AbstractTensor
from ..perturbation import PerturbationLinfNorm

MULTIPROCESS_MODEL = None
DEBUG = False
  
@torch.no_grad
@beartype
def build_solver_module(
    self, 
    x_L: torch.Tensor, 
    x_U: torch.Tensor, 
    C: None | torch.Tensor = None, 
    timeout: None | float | int = None, 
) -> None:
    # reset solver
    final = self.final_node()
    _reset_solver_vars(self, final)
    
    if DEBUG:
        print('[+] Build solver module:')
        print(f'{x_L=}')
        print(f'{x_U=}')
        print(f'{C=}')
        print(f'{timeout=}')
    
    # initialize solver
    self.solver_model = grb.Model('`MIP solver`')
    self.solver_model.setParam('OutputFlag', False)
    self.solver_model.setParam("FeasibilityTol", 1e-5)
    self.solver_model.setParam('MIPGap', 1e-2)  # Relative gap between lower and upper objective bound 
    self.solver_model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between lower and upper objective bound 

    # forward
    x = AbstractTensor(x_U, PerturbationLinfNorm(x_L=x_L, x_U=x_U)).to(self.device)
    self(x)
    
    # create interval ranges for input and other weight parameters
    roots = [self[name] for name in self.root_names]
    for i in range(len(roots)):
        value = roots[i].forward()
        if type(roots[i]) is AbstractInput:
            inp_gurobi_vars = _build_solver_input(self, roots[i])
            if DEBUG:
                print(f'Input: {inp_gurobi_vars.shape}')
        else:
            # regular weights/buffers
            roots[i].solver_vars = value
            
    # backward propagate every layer including last layer
    _build_solver_layer(self, x=x, node=final, C=C, timeout=timeout)
            
    # update final model
    self.solver_model.update()


@beartype
@torch.no_grad
def _reset_solver_vars(self, node: AbstractBase) -> None:
    if hasattr(node, 'solver_vars'):
        print(f'[+] delete solver_vars: {node=}')
        del node.solver_vars
    if hasattr(node, 'inputs'):
        for n in node.inputs:
            _reset_solver_vars(self, n)


@beartype
@torch.no_grad
def _build_solver_input(self, node: AbstractInput) -> np.ndarray:
    ## Do the input layer, which is a special case
    assert isinstance(node, AbstractInput)
    assert isinstance(node.perturbation, PerturbationLinfNorm), f'{node.perturbation=}'
    assert self.solver_model is not None
        
    # predefined vars will be shared within the solver model
    self.solver_model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    self.solver_model.addVar(lb=1, ub=1, obj=0, vtype=grb.GRB.CONTINUOUS, name='one')
    self.solver_model.addVar(lb=-1, ub=-1, obj=0, vtype=grb.GRB.CONTINUOUS, name='neg_one')
    
    # input bounds
    x_L = node.perturbation.x_L
    x_U = node.perturbation.x_U
    assert len(x_L) == len(x_U) == 1
    assert torch.all(x_L <= x_U)
    
    # input vars
    this_layer_shape = x_L.shape
    inp_gurobi_vars = []
    for dim, (lb, ub) in enumerate(zip(x_L.flatten(), x_U.flatten())):
        v = self.solver_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{dim}')
        inp_gurobi_vars.append(v)
    inp_gurobi_vars = np.array(inp_gurobi_vars).reshape(this_layer_shape)

    node.solver_vars = inp_gurobi_vars
    self.solver_model.update()
    return inp_gurobi_vars


@beartype
@torch.no_grad
def _build_solver_layer(
    self, 
    x: AbstractTensor, 
    node: AbstractBase, 
    C: torch.Tensor | None = None, 
    timeout: float | int | None = None, 
) -> np.ndarray | torch.Tensor:
    
    if hasattr(node, 'solver_vars'):
        return node.solver_vars
    
    for n in node.inputs:
        _build_solver_layer(self, x=x, node=n, C=C, timeout=timeout)
    
    if DEBUG:
        print(f'[+] build solver layer: {node=}')

    inp = [n_pre.solver_vars for n_pre in node.inputs]
    
    is_final_node = False
    if C is not None and isinstance(node, AbstractLinear) and self.final_name == node.name:
        # when node is the last layer, merge node with the specification, available when weights of this layer are not perturbed
        is_final_node = True
        solver_vars = node.build_solver(*inp, model=self.solver_model, C=C)
    else:
        solver_vars = node.build_solver(*inp, model=self.solver_model, C=None)
        
    # compute bounds for vars with "inf" bounds
    if not is_final_node:
        lower, upper = _optimize_layer_bound(node=node, model=self.solver_model, timeout_per_neuron=timeout)
        node.lower = lower.view(node.output_shape)
        node.upper = upper.view(node.output_shape)
    return solver_vars


@beartype
@torch.no_grad
def _optimize_layer_bound(
    node: AbstractBase, 
    model: grb.Model, 
    timeout_per_neuron: float | int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    global MULTIPROCESS_MODEL
    flattened_vars = np.array(node.solver_vars).reshape(-1)
    candidates = [v.VarName for v in flattened_vars if isinstance(v, grb.Var) and (v.lb == -float('inf') or v.ub == float('inf'))]

    if DEBUG:
        print(f'\t- optimize layer bound: {node=}')
        print(f'\t- layer vars: {node.solver_vars.shape=}')
        print(f'\t- {candidates=}')
        
    if len(candidates):
        # optimize bounds
        MULTIPROCESS_MODEL = model.copy()
        if timeout_per_neuron:
            MULTIPROCESS_MODEL.setParam('TimeLimit', timeout_per_neuron)
        max_worker = min(len(candidates), os.cpu_count() // 2)
        with multiprocessing.Pool(max_worker) as pool:
            solver_results = pool.map(_mip_solver_worker, candidates, chunksize=1)
        MULTIPROCESS_MODEL = None
        
        # update bounds        
        for (var_name, var_lb, var_ub) in solver_results:
            v = model.getVarByName(var_name)
            v.lb = max(v.lb, var_lb)
            v.ub = min(v.ub, var_ub)
            assert var_lb <= var_ub, f'{var_lb=:.06f} {var_ub=:.06f}'
        model.update()
            
    lower = torch.Tensor([v.lb for v in flattened_vars])
    upper = torch.Tensor([v.ub for v in flattened_vars])
    return lower, upper
  
@beartype
@torch.no_grad
def _mip_solver_worker(candidate: str) -> tuple[str, float, float]:
    def get_grb_solution(mip_model):
        status = mip_model.status
        if status == 9: # Timed out. Get current bound.
            bound = mip_model.objbound
        elif status == 2: # Optimally solved.
            bound = mip_model.objbound
        elif status == 15: # Early stop.
            raise NotImplementedError('Not supported yet')
        else:
            raise NotImplementedError(f'{status=}')
        return bound, status

    def solve_ub(model, v):
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        # model.setParam('BestBdStop', -eps)  # Terminiate as long as we find a negative upper bound.
        model.optimize()
        return get_grb_solution(model)

    def solve_lb(model, v):
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        # model.setParam('BestBdStop', eps)  # Terminiate as long as we find a positive lower bound.
        model.optimize()
        return get_grb_solution(model)

    start_time = time.time()
    model = MULTIPROCESS_MODEL.copy()
    var_name = candidate
    v = model.getVarByName(var_name)
    model.update()

    vlb, status_lb = solve_lb(model, v)
    vub, status_ub = solve_ub(model, v)
            
    if DEBUG:
        print(
            f"\t\t+ Solving MIP for {v.VarName:<10} (timeout={model.Params.TimeLimit}s): "
            f"[{v.lb:.6f}, {v.ub:.6f}]=>[{vlb:.6f}, {vub:.6f}] ({status_lb}, {status_ub}), "
            f"time: {time.time()-start_time:.4f}s, "
            f"#vars: {model.NumVars}, #constrs: {model.NumConstrs}"
        )
        sys.stdout.flush()

    return var_name, vlb, vub
