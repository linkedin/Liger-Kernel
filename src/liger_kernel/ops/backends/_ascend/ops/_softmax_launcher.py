"""A typed entry point into the Triton-Ascend 3.2.2 generated launcher."""

import ctypes
import importlib.util
import json

from triton.backends.ascend.driver import generate_npu_header_src
from triton.backends.ascend.driver import generate_npu_wrapper_src
from triton.backends.ascend.driver import make_npu_launcher_stub


def native_entry(compiled, grid):
    """Reuse backend packing with a typed launch and lightweight queue bridge."""
    src = compiled.src

    def index(key):
        return src.fn.arg_names.index(key) if isinstance(key, str) else key

    constants = {index(k): v for k, v in src.constants.items()}
    signature = {index(k): v for k, v in src.signature.items()}
    if len(signature) not in (2, 3) or any(not ty.startswith("*") for ty in signature.values()):
        raise RuntimeError("Unexpected Softmax runtime signature")
    wrapper = generate_npu_wrapper_src(constants, signature, compiled.metadata)
    # Keep the backend's packing/profiling logic and substitute only its queue bridge.
    start = wrapper.index("static void _launch(")
    end = wrapper.index("// Extract tensor shape", start)
    typed_launch = (
        wrapper[start:end]
        .replace("static void _launch(", "static void _liger_launch(liger_async_launch_t bridge, ", 1)
        .replace("g_async_launch", "bridge")
    )
    queue_call = "bridge(static_cast<void*>(&launch_call), name.c_str());"
    if typed_launch.count(queue_call) != 1:
        raise RuntimeError("Unexpected Ascend launcher queue invocation")
    typed_launch = typed_launch.replace(queue_call, "bridge(static_cast<void*>(&launch_call), name);")
    # The generated launcher and native extension both use torch's C++ ABI.
    wrapper += "\nusing liger_async_launch_t = void (*)(void*, const std::string&);\n"
    # The typed entry fixes the kernel name for this module. Static storage keeps
    # the queued callback from copying a heap-backed name on every invocation.
    name_init = 'std::string name = "";\n  name.append(kernelName);'
    if typed_launch.count(name_init) != 1:
        raise RuntimeError("Unexpected Ascend launcher name initialization")
    typed_launch = typed_launch.replace(name_init, "static const std::string name(kernelName);")
    fast_launch = typed_launch.replace("_liger_launch(", "_liger_launch_fast(", 1)
    fast_launch = fast_launch.replace("&tensorShapes", "&unusedShapes", 1).replace("&tensorKinds", "&unusedKinds", 1)
    fast_launch = fast_launch.replace(
        "static const std::string name(kernelName);",
        "static const std::string name(kernelName);\n"
        "  static const std::vector<std::vector<int64_t>> tensorShapes;\n"
        "  static const std::vector<int> tensorKinds;",
        1,
    )
    wrapper += typed_launch + fast_launch
    kinds = ", ".join(str(int(k)) for k in compiled.packed_metadata.get("tensor_kinds", []))
    args = ", ".join(f"reinterpret_cast<void*>(p{i})" for i in range(len(signature)))
    name = json.dumps(compiled.packed_metadata["kernel_name"])
    wrapper += f"""
extern "C" __attribute__((visibility("default"))) void liger_softmax_launch(
    uintptr_t function, uintptr_t stream, uintptr_t p0, uintptr_t p1, uintptr_t p2,
    int64_t rank, const int64_t* sizes, uintptr_t bridge) {{
  std::vector<std::vector<int64_t>> shapes;
  std::vector<int> kinds;
  if (__MsprofFlagL1) {{
    std::vector<int64_t> shape;
    if (rank) shape.assign(sizes, sizes + rank);
    shapes.assign({len(signature)}, shape);
    kinds = {{{kinds}}};
  }}
  auto launch = (__MsprofFlagL0 || __MsprofFlagL1) ? _liger_launch : _liger_launch_fast;
  launch(reinterpret_cast<liger_async_launch_t>(bridge), {name}, reinterpret_cast<const void*>(function),
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
    address = ctypes.cast(library.liger_softmax_launch, ctypes.c_void_p).value
    return (module, library), address
