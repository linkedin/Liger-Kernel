"""A typed entry point into the Triton-Ascend 3.2.2 generated launcher."""

import ctypes
import importlib.util
import json

from triton.backends.ascend.driver import generate_npu_header_src
from triton.backends.ascend.driver import generate_npu_wrapper_src
from triton.backends.ascend.driver import make_npu_launcher_stub


def native_entry(compiled, grid):
    """Keep backend launch logic intact, omitting only Python argument parsing."""
    src = compiled.src

    def index(key):
        return src.fn.arg_names.index(key) if isinstance(key, str) else key

    constants = {index(k): v for k, v in src.constants.items()}
    signature = {index(k): v for k, v in src.signature.items()}
    if len(signature) not in (2, 3) or any(not ty.startswith("*") for ty in signature.values()):
        raise RuntimeError("Unexpected ReLU squared runtime signature")
    wrapper = generate_npu_wrapper_src(constants, signature, compiled.metadata)
    kinds = ", ".join(str(int(k)) for k in compiled.packed_metadata.get("tensor_kinds", []))
    args = ", ".join(f"reinterpret_cast<void*>(p{i})" for i in range(len(signature)))
    name = json.dumps(compiled.packed_metadata["kernel_name"])
    wrapper += f"""
extern "C" __attribute__((visibility("default"))) void liger_relu_launch(
    uintptr_t function, uintptr_t stream, uintptr_t p0, uintptr_t p1, uintptr_t p2,
    int64_t rank, const int64_t* sizes) {{
  std::vector<std::vector<int64_t>> shapes;
  if (__MsprofFlagL1) {{
    std::vector<int64_t> shape;
    if (rank) shape.assign(sizes, sizes + rank);
    shapes.assign({len(signature)}, shape);
  }}
  std::vector<int> kinds = {{{kinds}}};
  _launch({name}, reinterpret_cast<const void*>(function),
          reinterpret_cast<rtStream_t>(stream), {grid[0]}, {grid[1]}, {grid[2]},
          shapes, kinds, {args});
}}
"""
    path = make_npu_launcher_stub(generate_npu_header_src(), wrapper, compiled.metadata.debug)
    # Execute the generated module initializer, including profiler registration.
    spec = importlib.util.spec_from_file_location("__triton_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = ctypes.CDLL(path)
    address = ctypes.cast(library.liger_relu_launch, ctypes.c_void_p).value
    return (module, library), address
