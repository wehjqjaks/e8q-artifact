#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_e8q(py::module_& m);
void register_me8q(py::module_& m);

PYBIND11_MODULE(_e8qlib, m) {
    m.doc() = "E8Q / ME8Q Python bindings";
    register_e8q(m);
    register_me8q(m);
}
