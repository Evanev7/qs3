"""Extract one fixed NVFP4 CuTe recipe from the pinned b12x source.

Generated source stays in the experiment output. This does not import b12x,
Torch, its planner, or its JIT at runtime, and does not edit the donor tree.
"""
import ast
import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / '.prototypes/kernel_replacements/vendor/b12x/b12x/_lib'
PIN = '0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68'
IMPORTS = '''from typing import Callable, List, Literal, Optional, Tuple, Type
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
'''


def extract(output: Path, tile_m: int, tile_n: int, tile_k: int, splits: int):
    source = (VENDOR / 'dense_gemm.py').read_text()
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'DenseGemmKernel')
    original_path = output / 'donor_class.py'
    original_path.write_text(IMPORTS + '\n' + ast.get_source_segment(source, cls))
    spec = importlib.util.spec_from_file_location('nvfp4_donor_class', original_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.DenseGemmKernel(
        sf_vec_size=16, mma_tiler_mn=(tile_m, tile_n), cluster_shape_mn=(1, 1),
        mma_k=64, tile_k=tile_k, split_k_slices=splits,
        split_k_atomic_bf16=False, use_m1_non_tma_c=False,
        fused_quant_bf16=False,
    )
    constants = {k:v for k,v in vars(instance).items()
                 if isinstance(v, (bool,int,float,str,tuple)) or v is None}
    # Attributes computed in _setup_attributes must remain mutable.
    for key in ['tiled_mma','ab_stage','epi_stage','a_smem_layout_staged',
                'b_smem_layout_staged','epi_smem_layout_staged']:
        constants.pop(key, None)
    constants.update(a_dtype=module.cutlass.Float4E2M1FN,
                     b_dtype=module.cutlass.Float4E2M1FN,
                     sf_dtype=module.cutlass.Float8E4M3FN,
                     c_dtype=module.cutlass.Float32 if splits>1 else module.cutlass.BFloat16,
                     acc_dtype=module.cutlass.Float32, packed_expand_ahead=False)

    class Fold(ast.NodeTransformer):
        def evaluate(self, node):
            if isinstance(node, ast.Call) and ast.unparse(node.func)=='cutlass.const_expr':
                node = node.args[0]
            return eval(compile(ast.Expression(node), '<fold>', 'eval'),
                        {'cutlass':module.cutlass, 'self':instance})

        def visit_Attribute(self, node):
            if (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
                    and node.value.id=='self' and node.attr in constants):
                value=constants[node.attr]
                expression='cutlass.'+value.__name__ if isinstance(value,type) else repr(value)
                return ast.copy_location(ast.parse(expression, mode='eval').body,node)
            return self.generic_visit(node)

        def visit_If(self, node):
            node=self.generic_visit(node)
            if not node.body and not node.orelse: return None
            if not node.body: node.body=[ast.Pass()]
            try: value=self.evaluate(node.test)
            except (NameError,AttributeError,TypeError,ValueError): return node
            return (node.body if value else node.orelse) if type(value) is bool else node

        def visit_IfExp(self,node):
            node=self.generic_visit(node)
            try: value=self.evaluate(node.test)
            except (NameError,AttributeError,TypeError,ValueError): return node
            return (node.body if value else node.orelse) if type(value) is bool else node

        def visit_Call(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.func,ast.Name) and node.func.id=='range'
                    and any(k.arg=='unroll' for k in node.keywords)):
                node.func=ast.Attribute(ast.Name('cutlass',ast.Load()),'range',ast.Load())
            return node

    cls=Fold().visit(copy.deepcopy(cls))
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)
             and n.name not in ('__init__','can_implement')}
    needed={'__call__'}
    while True:
        previous=needed.copy()
        for name in previous:
            for n in ast.walk(methods[name]):
                if (isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
                        and n.value.id in ('self','DenseGemmKernel') and n.attr in methods):
                    needed.add(n.attr)
        if previous==needed: break
    threads=instance.num_mma_warps*32
    init=ast.parse(f'''def __init__(self):
    self.mma_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads={threads})
    self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads={threads})
''').body[0]
    cls.body=[init]+[n for name,n in methods.items() if name in needed]
    helpers={}
    for filename in ['dense_gemm.py','intrinsics.py','utils.py']:
        for node in ast.parse((VENDOR/filename).read_text()).body:
            if isinstance(node,ast.FunctionDef):helpers.setdefault(node.name,node)
    selected={};scan=[cls]
    while scan:
        for node in ast.walk(scan.pop()):
            if isinstance(node,ast.Name) and node.id in helpers and node.id not in selected:
                selected[node.id]=helpers[node.id];scan.append(helpers[node.id])
    license=source[:source.index('# This file is ported')]
    text=license+f'\n# Extracted from b12x {PIN}.\n'+IMPORTS
    if any(name.startswith('sm120_make_smem_layout') for name in selected):
        text+='\n# Scale layout helpers, from utils.py:\n' + '\n'.join('# '+line for line in ast.get_docstring(ast.parse((VENDOR/'utils.py').read_text())).splitlines())+'\n'
    text+='\n\n'.join(ast.unparse(n) for n in selected.values())+'\n\n'+ast.unparse(cls)+'\n'
    text+='''
@cute.jit
def kernel(a: cute.Pointer, b: cute.Pointer, sfa: cute.Pointer, sfb: cute.Pointer,
           output: cute.Pointer, alpha: cute.Pointer, m: cutlass.Int32,
           stream: cuda.CUstream, N: cutlass.Constexpr, K: cutlass.Constexpr):
    a_tensor = cute.make_tensor(cute.recast_ptr(a, dtype=cutlass.Float4E2M1FN),
                               cute.make_ordered_layout((m, K, 1), order=(1, 0, 2)))
    b_tensor = cute.make_tensor(cute.recast_ptr(b, dtype=cutlass.Float4E2M1FN),
                               cute.make_ordered_layout((N, K, 1), order=(1, 0, 2)))
    sa = cute.make_tensor(sfa, cute.make_layout((1,)))
    sb = cute.make_tensor(sfb, cute.make_layout((1,)))
    c = cute.make_tensor(output, cute.make_ordered_layout((m, N, SPLITS), order=(1, 0, 2)))
    scale = cute.make_tensor(alpha, cute.make_layout((1,)))
    DenseGemmKernel()(a_tensor, a_tensor, a_tensor, a_tensor, b_tensor, sa, sb,
                      c, c, c, c, scale, 48, stream)
'''.replace('SPLITS',str(splits))
    path=output/f'nvfp4_m{tile_m}n{tile_n}k{tile_k}s{splits}.py'
    path.write_text(text)
    return path
