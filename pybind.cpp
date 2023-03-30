#include "justlm.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;



PYBIND11_MODULE(libjustlm_py, m) {
    using namespace LM;
    py::class_<Inference>(m, "Inference")
        .def(py::init<const std::string &, int32_t>(), py::arg("weights_path"), py::arg("seed") = 0)
        .def("append", &Inference::append, py::arg("prompt"), py::arg("on_tick") = nullptr)
        .def("run", &Inference::run, py::arg("end"), py::arg("on_tick") = nullptr)
        .def_readwrite("params", &Inference::params);
    py::class_<Inference::Params>(m, "Params")
        .def_readonly("seed", &Inference::Params::seed)
        .def_readwrite("n_threads", &Inference::Params::n_threads)
        .def_readonly("n_ctx", &Inference::Params::n_ctx)
        .def_readonly("n_prompt", &Inference::Params::n_prompt)
        .def_readwrite("n_batch", &Inference::Params::n_batch)
        .def_readwrite("top_k", &Inference::Params::top_k)
        .def_readwrite("top_p", &Inference::Params::top_p)
        .def_readwrite("temp", &Inference::Params::temp);
}
