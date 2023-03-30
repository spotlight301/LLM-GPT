#include "justlm.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;



PYBIND11_MODULE(libjustlm_py, m) {
    using namespace LM;
    py::class_<Inference>(m, "Inference")
        .def(py::init<const std::string &, int32_t>(), py::arg("weights_path"), py::arg("seed") = 0)
        .def("append", &Inference::append, py::arg("prompt"), py::arg("on_tick") = nullptr)
        .def("run", &Inference::run, py::arg("end"), py::arg("on_tick") = nullptr);
}
