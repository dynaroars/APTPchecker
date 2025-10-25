from collections import OrderedDict, namedtuple
from beartype import beartype
from packaging import version
import typing
import torch
import re


Node = namedtuple(
    'Node', 
    (
        'name', 'ori_name', 
        'inputs', 'attr', 'op', 'param', 
        'input_index', 'output_index',
        'bound_node', 'perturbation'
    ), 
    defaults=(None,) * 10
)

@beartype
def get_node_name(node: torch.Value) -> str:
    return node.debugName()

@beartype
def get_node_attribute(node: torch.Node, attribute_name: str):
    if hasattr(torch.onnx.symbolic_helper, '_node_get'): # Pytorch >= 1.13.
        return torch.onnx.symbolic_helper._node_get(node, attribute_name)
    else: # Pytorch <= 1.12. This will call _node_getitem in torch.onnx.utils.
        return node[attribute_name]


@beartype
def get_name_with_scope(scopes: dict, node: torch.Value) -> str:
    name = get_node_name(node)
    name = '/'.join([scopes[name], name])
    if '.' in name:
        # "." should not be used as it could issues in state_dict loading
        name = name.replace('.', '-')
    return name

@beartype
def get_output_template(out) -> None | dict:
    if isinstance(out, torch.Tensor):
        return None
    elif isinstance(out, list):
        return list([get_output_template(o) for o in out])
    elif isinstance(out, tuple):
        return tuple([get_output_template(o) for o in out])
    elif isinstance(out, dict):
        template = {}
        for key in out:
            template[key] = get_output_template(out[key])
        return template
    else:
        raise NotImplementedError


@beartype
def _get_jit_params(module, param_exclude: str | None, param_include: str | None) -> zip:
    state_dict = torch.jit._unique_state_dict(module, keep_vars=True)

    if param_exclude is not None:
        param_exclude = re.compile(param_exclude)
    if param_include is not None:
        param_include = re.compile(param_include)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if param_exclude is not None and param_exclude.match(k) is not None:
            print(f'\nremove input element {k} from nodesIn\n')
            continue
        if param_include is not None and param_include.match(k) is None:
            continue
        new_state_dict[k] = v

    params = zip(new_state_dict.keys(), new_state_dict.values())

    return params


@beartype
def _parse_inputs(graph: torch.Graph, inputs : tuple[torch.Tensor], params: tuple, scopes: dict) -> list[Node]:
    input_all = []
    for n in graph.inputs():
        input_all.append(n.debugName())
    
    input_used = []
    for n in graph.nodes():
        for inp in n.inputs():
            input_used.append(get_node_name(inp))
            
    for n in graph.outputs():
        name = get_node_name(n)
        if name in input_all:
            # This output node directly comes from an input node with an Op
            input_used.append(n.debugName())
            
    nodesIn = []
    used_by_index = []
    for i, n in enumerate(graph.inputs()):
        name = get_node_name(n)
        used = name in input_used
        used_by_index.append(used)
        if used:
            nodesIn.append(n)
    
    assert len(list(graph.inputs())) == len(inputs) + len(params)
    inputs = [inputs[i] for i in range(len(inputs)) if used_by_index[i]]
    input_index = [i for i in range(len(inputs)) if used_by_index[i]]
    inputs = list(zip(["input_{}".format(input_index[i]) for i in range(len(inputs))], inputs))
    
    params = [params[i] for i in range(len(params)) if used_by_index[i + len(inputs)]]
    inputs_and_params = inputs + params
    assert len(nodesIn) == len(inputs_and_params)
    
    # extract attributes
    for i, n in enumerate(nodesIn):
        if isinstance(inputs_and_params[i][1], torch.Tensor):
            perturbation = None
        else:
            raise NotImplementedError(f'Not supported type: {type(inputs_and_params[i][1])}')
        
        # check parameters
        if i >= len(inputs) and n.type().sizes() != list(inputs_and_params[i][1].size()):
            raise RuntimeError("Parameter shapes do not match: {} != {}".format(n.type().sizes(), list(inputs_and_params[i][1].size())))
        
        nodesIn[i] = Node(**{
            'op': 'Parameter',
            'name': get_name_with_scope(scopes, n),
            'inputs': [],
            'ori_name': inputs_and_params[i][0],
            'param': inputs_and_params[i][1] if i >= len(inputs) else None,
            'attr': str(n.type()),
            'input_index': input_index[i] if i < len(inputs) else None, # index among all the inputs including unused ones
            'perturbation': perturbation, 
        })
    
    return nodesIn
        
@beartype
def _parse_ops(graph: torch.Graph, scopes: dict) -> list[Node]:
    nodesOP = []
    for n in graph.nodes():
        attrs = {k: get_node_attribute(n, k) for k in n.attributeNames()}
        inps = [get_name_with_scope(scopes, i) for i in n.inputs()]
        for i, out in enumerate(list(n.outputs())):
            nodesOP.append(Node(**{
                'name': get_name_with_scope(scopes, out),
                'op': n.kind(),
                'inputs': inps,
                'attr': attrs,
                'output_index': i,
            }))
            # print(nodesOP[-1])
            
    return nodesOP

@beartype
def _parse_outputs(graph: torch.Graph, scopes: dict) -> list[str]:
    nodesOut = []
    for n in graph.outputs():
        nodesOut.append(get_name_with_scope(scopes, n))
        
    return nodesOut

@beartype
def parse_graph(graph: torch.Graph, inputs: tuple[torch.Tensor], params: tuple) -> tuple[list[Node], list[Node], list[str]]:
    # 1. scopes of the module
    scopes = {}
    for n in graph.nodes():
        for out in n.outputs():
            scopes[get_node_name(out)] = n.scopeName()
    for node in graph.inputs():
        name = get_node_name(node)
        scopes[name] = ''
        
    # 2. nodes of the module
    nodesIn = _parse_inputs(graph, inputs, params, scopes)
    nodesOP = _parse_ops(graph, scopes)
    nodesOut = _parse_outputs(graph, scopes)
    
    # 3. get parameters for ops
    for i in range(len(nodesOP)):
        param_in = OrderedDict()
        for inp in nodesOP[i].inputs:
            for n in nodesIn:
                if inp == n.name:
                    param_in.update({inp: n.param})
        nodesOP[i] = nodesOP[i]._replace(param=param_in)
    
    # debug
    # print('== nodesIn ==')
    # for n in nodesIn:
    #     print(f'\t- {n=}')
        
    # print('== nodesOP ==')
    # for n in nodesOP:
    #     print(f'\t- {n=}')

    # print('== nodesOut ==')
    # for n in nodesOut:
    #     print(f'\t- {n=}')
    # exit()
      
    return nodesOP, nodesIn, nodesOut

@beartype
def parse_module(module, inputs: torch.Tensor, param_exclude: str = ".*AuxLogits.*", param_include: str | None = None) -> tuple[list[Node], list[Node], list[str], typing.Any]:

    trace, out = torch.jit._get_trace_graph(module, inputs)
    # print(f'\n\n[+] trace:\n{trace}')

    if version.parse(torch.__version__) < version.parse("2.0.0"):
        from torch.onnx.symbolic_helper import _set_opset_version
        _set_opset_version(12)
        
    primary_input = get_node_name(next(iter(trace.inputs())))
    # print(f'\n\n[+] {primary_input=}')
    
    trace_graph = torch.onnx.utils._optimize_graph(
        graph=trace, 
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        params_dict={},
        input_names=[primary_input],
        dynamic_axes={primary_input: {0: 'batch'}}
    )
    # print(f'\n\n[+] trace_graph:\n{trace_graph}')
    
    params = _get_jit_params(
        module=module, 
        param_exclude=param_exclude, 
        param_include=param_include,
    )
          
    # extract graph
    nodesOP, nodesIn, nodesOut = parse_graph(trace_graph, tuple(inputs), tuple(params))
    template = get_output_template(out)
    # print(f'{template=}')
    
    return nodesOP, nodesIn, nodesOut, template
