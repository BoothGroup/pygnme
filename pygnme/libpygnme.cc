#include <string>
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <carma>
#include <armadillo>

#include "wick_orbitals.h"
#include "wick_rscf.h"
#include "wick_uscf.h"
#include "bitset.h"
#include "bitset_tools.h"

#include "wick_api.h"
#include "utils_api.h"

namespace py = pybind11;

namespace pybind11::literals {

using Complex = std::complex<double>;


PYBIND11_MODULE(libpygnme, m) {
    m.attr("__name__") = "libpygnme";
    m.doc() = "pybind11 interface to libgnme";

    // TODO these are experimental
    //m.def("field_to_array", [](const arma::field<double> &f) -> py::array_t<double> { return field_to_array(f); });
    //m.def("array_to_field", [](const py::array_t<double> &arr) -> arma::field<double> { return array_to_field(arr); });


    // pygnme.wick

    py::module wick = m.def_submodule("wick");

    export_wick_orbitals<double, double>(wick, "double_double");
    export_wick_orbitals<Complex, double>(wick, "complex_double");
    export_wick_orbitals<Complex, Complex>(wick, "complex_complex");

    export_wick_rscf<double, double, double>(wick, "double_double_double");
    export_wick_rscf<Complex, double, double>(wick, "complex_double_double");
    export_wick_rscf<Complex, Complex, double>(wick, "complex_complex_double");
    export_wick_rscf<Complex, Complex, Complex>(wick, "complex_complex_complex");

    export_wick_uscf<double, double, double>(wick, "double_double_double");
    export_wick_uscf<Complex, double, double>(wick, "complex_double_double");
    export_wick_uscf<Complex, Complex, double>(wick, "complex_complex_double");
    export_wick_uscf<Complex, Complex, Complex>(wick, "complex_complex_complex");


    // pygnme.utils

    py::module utils = m.def_submodule("utils");

    export_bitset(utils);
    export_bitset_tools(utils);
}

}  // namespace pybind11:literals
