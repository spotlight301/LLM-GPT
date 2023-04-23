#include "justlm.hpp"
#include "justlm_pool.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;



PYBIND11_MODULE(libjustlm_py, m) {
    using namespace LM;
    py::class_<Inference::Params>(m, "Params")
        .def(py::init<>())
        .def_readonly("seed", &Inference::Params::seed)
        .def_readwrite("n_threads", &Inference::Params::n_threads)
        .def_readonly("n_ctx", &Inference::Params::n_ctx)
        .def_readonly("n_prompt", &Inference::Params::n_prompt)
        .def_readwrite("n_batch", &Inference::Params::n_batch)
        .def_readwrite("n_repeat_last", &Inference::Params::n_repeat_last)
        .def_readwrite("repeat_penalty", &Inference::Params::repeat_penalty)
        .def_readwrite("top_k", &Inference::Params::top_k)
        .def_readwrite("top_p", &Inference::Params::top_p)
        .def_readwrite("temp", &Inference::Params::temp)
        .def_readwrite("repeat_penalty", &Inference::Params::repeat_penalty)
        .def_readwrite("eos_ignores", &Inference::Params::eos_ignores)
        .def_readwrite("use_mlock", &Inference::Params::use_mlock);
    py::class_<Inference>(m, "Inference")
        .def(py::init<const std::string &, const Inference::Params&>(), py::arg("weights_path"), py::arg("params") = Inference::Params())
        .def("append", &Inference::append, py::arg("prompt"), py::arg("on_tick") = nullptr)
        .def("run", &Inference::run, py::arg("end") = "", py::arg("on_tick") = nullptr)
        .def("create_savestate", &Inference::create_savestate)
        .def("restore_savestate", &Inference::restore_savestate)
        .def("get_prompt", &Inference::get_prompt)
        .def_readwrite("params", &Inference::params);
    py::class_<Inference::Savestate>(m, "Savestate")
        .def(py::init<>());

    py::class_<InferencePool>(m, "InferencePool")
        .def(py::init<size_t, const std::string&, bool>(), py::arg("size"), py::arg("pool_name"), py::arg("clean_up") = true)
        .def("create_inference", &InferencePool::create_inference, py::arg("id"), py::arg("weights_path"), py::arg("parameters"), py::return_value_policy::reference_internal)
        .def("get_inference", &InferencePool::get_inference, py::arg("id"), py::return_value_policy::reference_internal)
        .def("get_or_create_inference", &InferencePool::create_inference, py::arg("id"), py::arg("weights_path"), py::arg("parameters"), py::return_value_policy::reference_internal)
        .def("delete_inference", &InferencePool::delete_inference, py::arg("id"))
        .def("store_all", &InferencePool::store_all)
        .def("set_store_on_destruct", &InferencePool::set_store_on_destruct)
        .def("is_stored_on_destruction", &InferencePool::is_stored_on_destruction)
        .def("get_active_slot_ids", &InferencePool::get_active_slot_ids);
}
