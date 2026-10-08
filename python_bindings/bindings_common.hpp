#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace e8qlib::python_bindings {

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_2d_array(
    py::handle value, const char* name
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D NumPy array");
    }
    return array;
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_1d_array(
    py::handle value, const char* name
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
    }
    return array;
}

}  // namespace e8qlib::python_bindings
