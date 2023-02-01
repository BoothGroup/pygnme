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
        // Constructors:
        .def(py::init<const size_t, const size_t, const size_t,
                      arma::Mat<Tc>, arma::Mat<Tc>, arma::Mat<Tb>>())
        .def(py::init<const size_t, const size_t, const size_t,
                      arma::Mat<Tc>, arma::Mat<Tc>, arma::Mat<Tb>,
                      const size_t, const size_t>())
        // Variables:
        .def_readonly("m_nbsf", &WickOrbitals::m_nbsf)
        .def_readonly("m_nmo", &WickOrbitals::m_nbsf)
        .def_readonly("m_nelec", &WickOrbitals::m_nbsf)
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
        .def_readwrite("m_Q", &WickOrbitals::m_Q)
        // m_metric is a reference:
        .def("m_metric", [](const WickOrbitals &c) { return c.m_metric; });
}

/*
 * Export the wick_rscf class
 */
template<typename Tc, typename Tf, typename Tb>
void export_wick_rscf(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("wick_rscf_") + typestr;
    using WickRscf = libgnme::wick_rscf<Tc, Tf, Tb>;
    using Bitset = libgnme::bitset;
    py::class_<WickRscf>(m, pyclass_name.c_str())
        // Constructors:
        .def(py::init<libgnme::wick_orbitals<Tc, Tb> &, const arma::Mat<Tb> &>())
        .def(py::init<libgnme::wick_orbitals<Tc, Tb> &, const arma::Mat<Tb> &, double>())
        // Variables:
        .def_readwrite("m_nz", &WickRscf::m_nz)
        // Functions:
        .def("add_one_body", &WickRscf::add_one_body)
        .def("add_two_body", &WickRscf::add_two_body)
        .def("evaluate_overlap", &WickRscf::evaluate_overlap)
        .def("evaluate_one_body_spin", &WickRscf::evaluate_one_body_spin)
        .def("evaluate_rdm1", &WickRscf::evaluate_rdm1)
        // evaluate requires overloading and return values since Tc will be
        // immutable on the python side
        .def("evaluate", 
                [](WickRscf &scf, Bitset &bxa, Bitset &bxb, Bitset &bwa, Bitset &bwb, Tc &S, Tc &V) {
                    scf.evaluate(bxa, bxb, bwa, bwb, S, V);
                    return std::make_tuple(S, V);
                }
        )
        .def("evaluate", 
                [](WickRscf &scf, arma::umat &xa_hp, arma::umat &xb_hp, arma::umat &wa_hp, arma::umat &wb_hp, Tc &S, Tc &V) {
                    scf.evaluate(xa_hp, xb_hp, wa_hp, wb_hp, S, V);
                    return std::make_tuple(S, V);
                }
        );
}

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

PYBIND11_MODULE(pygnme, m) {
    m.attr("__name__") = "pgnme";
    m.doc() = "pybind11 interface to libgnme";

    // TODO these are experimental
    m.def("field_to_array", [](const arma::field<double> &f) -> py::array_t<double> { return field_to_array(f); });
    m.def("array_to_field", [](const py::array_t<double> &arr) -> arma::field<double> { return array_to_field(arr); });


    // pygnme.wick

    py::module wick = m.def_submodule("wick");

    export_wick_orbitals<double, double>(wick, "double_double");
    export_wick_orbitals<Complex, double>(wick, "complex_double");
    export_wick_orbitals<Complex, Complex>(wick, "complex_complex");

    export_wick_rscf<double, double, double>(wick, "double_double_double");
    export_wick_rscf<Complex, double, double>(wick, "complex_double_double");
    export_wick_rscf<Complex, Complex, double>(wick, "complex_complex_double");
    export_wick_rscf<Complex, Complex, Complex>(wick, "complex_complex_complex");


    // pygnme.utils

    py::module utils = m.def_submodule("utils");

    export_bitset(utils);
    export_bitset_tools(utils);
}

}  // namespace pybind11:literals
