#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <carma>
#include <armadillo>

#include "bitset.h"
#include "bitset_tools.h"

namespace py = pybind11;

namespace pybind11::literals {

/*
 * Export the bitset class
 */
void export_bitset(py::module &m) {
    using Bitset = libgnme::bitset;
    py::class_<Bitset>(m, "bitset")
        .def(py::init())
        .def(py::init<const bitset &>())
        .def(py::init<std::vector<bool>())
        .def(py::init<int, int>())
        .def("flip", &bitset::flip)
        .def("print", &bitset::print)
        .def("count", &bitset::count)
        .def("get_int", &bitset::get_int)
        .def("excitation", &bitset::excitation)
        .def("occ", &bitset::occ)
        .def("next_fci", &bitset::next_fci);
}

/*
 * Export bitset_tools
 */
void export_bitset_tools(py::module &m) {
    m.def("fci_bitset_list", fci_bitset_list);
}

PYBIND11_MODULE(utils, m) {
    m.attr("__name__") = "pygnme.utils";
    m.doc() = "pybind11 interface to libgnme.utils";

    export_bitset(m);
    export_bitset_tools(m);
}

}  // namespace pybind11:literals
