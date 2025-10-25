from ..operator import *

abstract_op_mapping = {
    'onnx::Gemm': AbstractLinear,
    # 'prim::Constant': BoundPrimConstant,
    # 'custom::Gelu': BoundGelu,
    # 'onnx::Clip': BoundHardTanh,
}
