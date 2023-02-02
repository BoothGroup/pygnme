#ifndef PYGNME_UTILS_API_H_
#define PYGNME_UTILS_API_H_

#include <string>
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <carma>
#include <armadillo>

#include "wick_orbitals.h"
#include "wick_rscf.h"
#include "bitset.h"
#include "bitset_tools.h"

namespace py = pybind11;

namespace pybind11::literals {

using Complex = std::complex<double>;


/*
 * Export the bitset class
 */
void export_bitset(py::module &m) {
    using Bitset = libgnme::bitset;
    py::class_<Bitset>(m, "bitset")
        .def(py::init())
        .def(py::init<const Bitset &>())
        .def(py::init<std::vector<bool>>())
        .def(py::init<int, int>())
        .def("flip", &Bitset::flip)
        .def("print", &Bitset::print)
        .def("count", &Bitset::count)
        .def("get_int", &Bitset::get_int)
        .def("excitation", &Bitset::excitation)
        .def("occ", &Bitset::occ)
        .def("next_fci", &Bitset::next_fci);
}

/*
 * Export bitset_tools
 */
void export_bitset_tools(py::module &m) {
    m.def("fci_bitset_list", libgnme::fci_bitset_list);
}

}  // namespace pybind11:literals

#endif
