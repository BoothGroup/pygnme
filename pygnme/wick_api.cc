#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <carma>
#include <armadillo>

#include "wick_orbitals.h"
#include "wick_rscf.h"
#include "bitset.h"

namespace py = pybind11;

namespace pybind11::literals {

/*
 * Support for arma::field
 */
py::array_t<double> field_to_array(const arma::field<double> &f) {
    py::array_t<double> arr({f.n_elem, f.n_rows, f.n_cols});
    for (int i = 0; i < f.n_elem; i++) {
        //arr.mutable_unchecked<3>()(i, arma::span(), arma::span()) = f(i);
        for (int j = 0; j < f.n_rows; j++) {
            for (int k = 0; k < f.n_cols; k++) {
                arr.mutable_unchecked<3>()(i, j, k) = f(i);
            }
        }
    }
    return arr;
}
arma::field<double> array_to_field(const py::array_t<double> &arr) {
    auto dims = arr.ndim();
    if (dims != 3) {
        throw std::invalid_argument("Input must have 3 dimensions");
    }
 
    const auto n_elem = arr.shape(0);
    const auto n_rows = arr.shape(1);
    const auto n_cols = arr.shape(2);
 
    arma::field<double> f(n_elem, n_rows, n_cols);
    for (int i = 0; i < n_elem; i++) {
        //f(i) = arr.unchecked<3>()(i, arma::span(), arma::span());
        for (int j = 0; j < n_rows; j++) {
            for (int k = 0; k < n_cols; k++) {
                f(i, j, k) = arr.unchecked<3>()(i, j, k);
            }
        }
    }
 
    return f;
}

/*
 * Export the wick_orbitals class
 */
template<typename Tc, typename Tb>
void export_wick_orbitals(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("wick_orbitals_") + typestr;
    using WickOrbitals = libgnme::wick_orbitals<Tc, Tb>;
    py::class_<WickOrbitals>(m, pyclass_name.c_str())
        .def(py::init<const size_t, const size_t, const size_t, arma::Mat<Tc>, arma::Mat<Tc>, arma::Mat<Tb>>())
        .def(py::init<const size_t, const size_t, const size_t, arma::Mat<Tc>, arma::Mat<Tc>, arma::Mat<Tb>, const size_t, const size_t>())
        .def_readonly("m_nbsf", &WickOrbitals::m_nbsf)
        .def_readonly("m_nmo", &WickOrbitals::m_nbsf)
        .def_readonly("m_nelec", &WickOrbitals::m_nbsf)
        .def("m_metric", [](const WickOrbitals &c) { return c.m_metric; })
        .def_readwrite("m_nact", &WickOrbitals::m_nact)
        .def_readwrite("m_ncore", &WickOrbitals::m_ncore)
        .def_readwrite("m_nz", &WickOrbitals::m_nz)
        .def_readwrite("m_redS", &WickOrbitals::m_redS)
        .def_readwrite("m_M", &WickOrbitals::m_M)
        .def_readwrite("m_X", &WickOrbitals::m_X)
        .def_readwrite("m_Y", &WickOrbitals::m_Y)
        .def_readwrite("m_CX", &WickOrbitals::m_CX)
        .def_readwrite("m_XC", &WickOrbitals::m_XC)
        .def_readwrite("m_wxP", &WickOrbitals::m_wxP)
        .def_readwrite("m_R", &WickOrbitals::m_R)
        .def_readwrite("m_Q", &WickOrbitals::m_Q);
}

/*
 * Export the wick_rscf class
 */
template<typename Tc, typename Tf, typename Tb>
void export_wick_rscf(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("wick_rscf_") + typestr;
    using WickRscf = libgnme::wick_rscf<Tc, Tf, Tb>;
    py::class_<WickRscf>(m, pyclass_name.c_str())
        .def(py::init<libgnme::wick_orbitals<Tc, Tb> &, const arma::Mat<Tb> &>())
        .def(py::init<libgnme::wick_orbitals<Tc, Tb> &, const arma::Mat<Tb> &, double>())
        .def_readwrite("m_nz", &WickRscf::m_nz)
        .def("add_one_body", &WickRscf::add_one_body)
        .def("add_two_body", &WickRscf::add_two_body)
        .def("evaluate_overlap", &WickRscf::evaluate_overlap)
        .def("evaluate_one_body_spin", &WickRscf::evaluate_one_body_spin)
        .def("evaluate_rdm1", &WickRscf::evaluate_rdm1)
        .def("evaluate", py::overload_cast<libgnme::bitset &, libgnme::bitset &, libgnme::bitset &, libgnme::bitset &, Tc &, Tc &>(&WickRscf::evaluate))
        .def("evaluate", py::overload_cast<arma::umat &, arma::umat &, arma::umat &, arma::umat &, Tc &, Tc &>(&WickRscf::evaluate));
}

PYBIND11_MODULE(wick, m) {
    m.attr("__name__") = "pygnme.wick";
    m.doc() = "pybind11 interface to libgnme.wick";

    m.def("field_to_array", [](const arma::field<double> &f) -> py::array_t<double> { return field_to_array(f); });
    m.def("array_to_field", [](const py::array_t<double> &arr) -> arma::field<double> { return array_to_field(arr); });

    // TODO can we generalise templates for the pybind11 layer?
    export_wick_orbitals<double, double>(m, "double_double");

    export_wick_rscf<double, double, double>(m, "double_double_double");
}

}  // namespace pybind11:literals
