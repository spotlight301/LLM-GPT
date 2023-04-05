#include "justlm.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;



PYBIND11_MODULE(libjustlm_py, m) {
    using namespace LM;
    py::class_<Inference>(m, "Inference")
        .def(py::init<const std::string &, const Inference::Params&>(), py::arg("weights_path"), py::arg("params") = 0)
        .def("append", &Inference::append, py::arg("prompt"), py::arg("on_tick") = nullptr)
        .def("run", &Inference::run, py::arg("end"), py::arg("on_tick") = nullptr)
        .def_readwrite("params", &Inference::params);
    py::class_<Inference::Params>(m, "Params")
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
}
