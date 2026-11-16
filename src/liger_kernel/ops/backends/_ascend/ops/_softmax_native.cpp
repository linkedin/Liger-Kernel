#include <torch/extension.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

namespace py = pybind11;
using torch::autograd::variable_list;
namespace {
constexpr size_t kMaxContexts = 128;
uintptr_t address(const at::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr());
}

// Resolve invariant compiled-kernel attributes once, but read streams and hooks
// per invocation. Tensor profiling and compile-only use NPULauncher; the
// ordinary path mirrors its registration update after the compiled launch.
struct Launcher {
  py::object compiled, grid, stream_fn, run, function, metadata, metadata_fn, kernel_type;
  using NativeEntry = void (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                                uintptr_t, int64_t, const int64_t*);
  py::object enter_key, exit_key, registered_key, builtin, backend_utils, native_library;
  NativeEntry native;
  uintptr_t function_address;
  int device;
  bool tensor_args, direct_launch;

  explicit Launcher(const py::tuple& spec)
      : compiled(spec[0]), grid(spec[1]), stream_fn(spec[2]), device(spec[3].cast<int>()) {
    native_library = spec[4];
    native = reinterpret_cast<NativeEntry>(spec[5].cast<uintptr_t>());
    run = compiled.attr("run");
    function = compiled.attr("function");
    function_address = function.cast<uintptr_t>();
    metadata = compiled.attr("packed_metadata");
    metadata_fn = compiled.attr("launch_metadata");
    kernel_type = py::type::of(compiled);
    tensor_args = run.attr("enable_msprof_register_tensor").cast<bool>();
    enter_key = py::str("launch_enter_hook");
    exit_key = py::str("launch_exit_hook");
    registered_key = py::str("TRITON_PROFILER_REGISTERED");
    direct_launch = !tensor_args && !run.attr("compile_only").cast<bool>();
    if (direct_launch) {
      builtin = run.attr("launch");
      backend_utils = py::module_::import("triton.backends.ascend.utils");
    }
  }

  ~Launcher() {
    if (Py_IsInitialized()) {
      py::gil_scoped_acquire gil;
      compiled = {}; grid = {}; stream_fn = {}; run = {};
      function = {}; metadata = {}; metadata_fn = {}; kernel_type = {};
      enter_key = {}; exit_key = {}; registered_key = {}; builtin = {}; backend_utils = {}; native_library = {};
    } else {
      compiled.release(); grid.release(); stream_fn.release(); run.release();
      function.release(); metadata.release(); metadata_fn.release(); kernel_type.release();
      enter_key.release(); exit_key.release(); registered_key.release(); builtin.release(); backend_utils.release(); native_library.release();
    }
  }

  template <typename... Args> void launch(Args&&... args) {
    auto stream = stream_fn(device);
    auto enter = kernel_type.attr(enter_key);
    auto exit = kernel_type.attr(exit_key);
    py::object launch_metadata = enter.is_none() ? py::none() : metadata_fn(grid, stream, args...);
    auto dims = grid.cast<py::tuple>();
    if (direct_launch) {
      auto registered = builtin(dims[0], dims[1], dims[2], stream, function, metadata,
                                launch_metadata, enter, exit, args...);
      int equal = PyObject_RichCompareBool(registered.ptr(), py::int_(1).ptr(), Py_EQ);
      if (equal < 0) throw py::error_already_set();
      backend_utils.attr("TRITON_PROFILER_REGISTERED") = equal == 1;
    } else {
      run(dims[0], dims[1], dims[2], stream, function, metadata, launch_metadata,
          enter, exit, args...);
    }
  }

  bool native_launch(const at::Tensor& x, uintptr_t p1, uintptr_t p2) {
    if (!native || !direct_launch) return false;
    // Read the actual class dictionary under the GIL on every invocation.
    // Missing/inherited hooks use the general lookup path; dynamic assignment
    // after forward remains visible without allocating temporary py::objects.
    auto dict = reinterpret_cast<PyTypeObject*>(kernel_type.ptr())->tp_dict;
    if (PyDict_GetItemWithError(dict, enter_key.ptr()) != Py_None ||
        PyDict_GetItemWithError(dict, exit_key.ptr()) != Py_None) {
      if (PyErr_Occurred()) throw py::error_already_set();
      return false;
    }
    auto stream = reinterpret_cast<uintptr_t>(
        c10_npu::getCurrentNPUStream(device).stream(false));
    native(function_address, stream, address(x), p1, p2, x.dim(), x.sizes().data());
    if (PyErr_Occurred()) throw py::error_already_set();
    // This generated entry has the same void result as the backend wrapper.
    auto globals = PyModule_GetDict(backend_utils.ptr());
    if (PyDict_GetItemWithError(globals, registered_key.ptr()) != Py_False &&
        PyDict_SetItem(globals, registered_key.ptr(), Py_False) < 0)
      throw py::error_already_set();
    return true;
  }

  void forward(const at::Tensor& x, const at::Tensor& out) {
    if (!native_launch(x, address(out), 0)) launch(x, out);
  }

  void backward(const at::Tensor& x, const at::Tensor& dy, const at::Tensor& dx) {
    if (!native_launch(dy, address(x), address(dx))) launch(dy, x, dx);
  }

};

struct SoftmaxBackward : torch::autograd::Node {
  torch::autograd::SavedVariable input;
  std::shared_ptr<Launcher> launcher;

  SoftmaxBackward(std::shared_ptr<Launcher> bwd)
      : launcher(std::move(bwd)) {}

  std::string name() const override { return "LigerSoftmaxBackward"; }

  variable_list apply(variable_list&& grads) override {
    if (!grads[0].defined()) return {at::Tensor()};
    at::Tensor x;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      x = input.unpack(shared_from_this());
    }
    TORCH_CHECK(grads[0].device() == x.device() &&
                grads[0].scalar_type() == x.scalar_type() &&
                grads[0].numel() == x.numel(), "Softmax gradient metadata mismatch");
    c10::DeviceGuard device_guard(x.device());
    auto dy = grads[0].contiguous();
    auto dx = at::empty(dy.sizes(), dy.options());
    if (x.numel()) {
      py::gil_scoped_acquire gil;
      launcher->backward(x, dy, dx);
    }
    return {dx};
  }

  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input.reset_data();
  }
};

struct SoftmaxContext {
  std::shared_ptr<Launcher> forward, backward;
  at::ScalarType dtype;
  int device;
  int64_t elements, columns;

  SoftmaxContext(const py::tuple& spec, const at::Tensor& x)
      : dtype(x.scalar_type()), device(x.device().index()), elements(x.numel()), columns(x.size(-1)) {
    if (elements) {
      forward = std::make_shared<Launcher>(spec[0].cast<py::tuple>());
      backward = std::make_shared<Launcher>(spec[1].cast<py::tuple>());
    }
  }

  at::Tensor apply(const at::Tensor& x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
                x.device().index() == device && x.scalar_type() == dtype &&
                x.numel() == elements && x.size(-1) == columns, "Softmax input metadata mismatch");
    bool needs_grad = torch::autograd::compute_requires_grad(x);
    c10::DeviceGuard device_guard(x.device());
    at::AutoGradMode guard(false);
    auto input = x.contiguous();
    std::shared_ptr<SoftmaxBackward> node;
    if (needs_grad) {
      node = std::make_shared<SoftmaxBackward>(backward);
      node->set_next_edges(torch::autograd::collect_next_edges(x));
    }
    auto out = at::empty(input.sizes(), input.options());
    if (elements) forward->forward(input, out);
    if (node) {
      torch::autograd::set_history(out, node);
      node->input = torch::autograd::SavedVariable(out, true);
    }
    return out;
  }
};

// GIL serializes cache access. The bounded cache holds compiled state, not
// request tensors; live autograd nodes retain only the backward launcher.
struct CachedOp {
  using Key = std::tuple<int64_t, int64_t, at::ScalarType, int>;
  using Entry = std::pair<Key, std::shared_ptr<SoftmaxContext>>;
  py::object factory;
  std::list<Entry> lru;
  std::map<Key, std::list<Entry>::iterator> entries;

  explicit CachedOp(py::object make) : factory(std::move(make)) {}

  std::shared_ptr<SoftmaxContext> context(const at::Tensor& x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "Softmax requires an NPU tensor");
    Key key{x.numel(), x.size(-1), x.scalar_type(), x.device().index()};
    auto found = entries.find(key);
    if (found == entries.end()) {
      auto spec = x.numel() ? factory(x).cast<py::tuple>() : py::tuple();
      auto created = std::make_shared<SoftmaxContext>(spec, x);
      // Compilation can release the GIL; another thread may have inserted it.
      found = entries.find(key);
      if (found == entries.end()) {
        lru.emplace_front(key, std::move(created));
        found = entries.emplace(key, lru.begin()).first;
        if (entries.size() > kMaxContexts) {
          entries.erase(lru.back().first);
          lru.pop_back();
        }
      }
    }
    lru.splice(lru.begin(), lru, found->second);
    return found->second->second;
  }

  at::Tensor apply(const at::Tensor& x) { return context(x)->apply(x); }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // torch_npu 2.9.0 lacks the Python NoWait binding. The generated launcher
  // already uses its task queue, so reading the stream need not drain it.
  m.def("current_stream", [](int device) {
    return reinterpret_cast<uintptr_t>(c10_npu::getCurrentNPUStream(device).stream(false));
  });
  py::class_<SoftmaxContext, std::shared_ptr<SoftmaxContext>>(m, "Context")
      .def("apply", &SoftmaxContext::apply);
  py::class_<CachedOp>(m, "CachedOp")
      .def(py::init<py::object>()).def("apply", &CachedOp::apply)
      .def("context", &CachedOp::context);
}
