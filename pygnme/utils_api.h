#ifndef PYGNME_UTILS_API_H_
#define PYGNME_UTILS_API_H_

#include <string>
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <carma>
#include <armadillo>

#include "libgnme/wick/wick_orbitals.h"
#include "libgnme/wick/wick_rscf.h"
#include "libgnme/utils/utils.h"
#include "libgnme/utils/bitset.h"
#include "libgnme/utils/bitset_tools.h"
#include "libgnme/utils/lowdin_pair.h"
#include "libgnme/utils/linalg.h"
#include "libgnme/utils/eri_ao2mo.h"

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
        // Functions:
        .def("flip", &Bitset::flip)
        .def("print", &Bitset::print)
        .def("count", &Bitset::count)
        .def("get_int", &Bitset::get_int)
        .def("excitation", &Bitset::excitation)
        .def("occ", &Bitset::occ)
        .def("next_fci", &Bitset::next_fci)
        .def("parity", [](Bitset &self, Bitset &other) {
                return self.parity(other);
            },
            py::arg("other")
        );
}

/*
 * Export bitset_tools
 */
void export_bitset_tools(py::module &m) {
    m.def("fci_bitset_list", libgnme::fci_bitset_list);
}

/*
 * Export noci_density
 */
void export_noci_density(py::module &m) {
    using Complex = std::complex<double>;
    m.def("rscf_noci_density", libgnme::rscf_noci_density<double, double>);
    m.def("rscf_noci_density", libgnme::rscf_noci_density<Complex, double>);
    m.def("uscf_noci_density",
            py::overload_cast<arma::Cube<double>, const arma::Col<double>, const arma::Mat<double>,
                              const size_t, const size_t, const size_t, const size_t, const size_t,
                              arma::Mat<double> &>(
                                  libgnme::uscf_noci_density<double, double>));
    m.def("uscf_noci_density",
            py::overload_cast<arma::Cube<double>, const arma::Col<double>, const arma::Mat<double>,
                              const size_t, const size_t, const size_t, const size_t, const size_t,
                              arma::Mat<double> &, arma::Mat<double> &>(
                                  libgnme::uscf_noci_density<double, double>));
    m.def("uscf_noci_density",
            py::overload_cast<arma::Cube<Complex>, const arma::Col<Complex>, const arma::Mat<double>,
                              const size_t, const size_t, const size_t, const size_t, const size_t,
                              arma::Mat<Complex> &>(
                                  libgnme::uscf_noci_density<Complex, double>));
    m.def("uscf_noci_density",
            py::overload_cast<arma::Cube<Complex>, const arma::Col<Complex>, const arma::Mat<double>,
                              const size_t, const size_t, const size_t, const size_t, const size_t,
                              arma::Mat<Complex> &, arma::Mat<Complex> &>(
                                  libgnme::uscf_noci_density<Complex, double>));
    m.def("gscf_noci_density", libgnme::gscf_noci_density<double, double>);
    m.def("gscf_noci_density", libgnme::gscf_noci_density<Complex, double>);
}

/*
 * Export lowdin_pair
 */
void export_lowdin_pair(py::module &m) {
    using Complex = std::complex<double>;
    m.def("lowdin_pair", libgnme::lowdin_pair<double, double>);
    m.def("lowdin_pair", libgnme::lowdin_pair<Complex, double>);
    m.def("lowdin_pair", libgnme::lowdin_pair<Complex, Complex>);
    m.def("reduced_overlap", libgnme::reduced_overlap<double>);
    m.def("reduced_overlap", libgnme::reduced_overlap<Complex>);
}

/*
 * Export linalg
 */
void export_linalg(py::module &m) {
    using Complex = std::complex<double>;
    m.def("orthogonalisation_matrix", libgnme::orthogonalisation_matrix<double>);
    m.def("orthogonalisation_matrix", libgnme::orthogonalisation_matrix<Complex>);
    m.def("gen_eig_sym",  [](
                          const size_t dim, 
                          arma::Mat<double> &M, arma::Mat<double> &S,  
                          double thresh) {
                arma::Mat<double> X, eigvec;
                arma::Col<double> eigval;
                libgnme::gen_eig_sym<double>(dim, M, S, X, eigval, eigvec, thresh);
                arma::Row<double> roweig = eigval.st();
                return std::make_tuple(roweig, eigvec);
          },
          py::arg("dim"),
          py::arg("M"),
          py::arg("S"),
          py::arg("thresh") = 1e-8
    );
    m.def("gen_eig_sym",  [](
                          const size_t dim, 
                          arma::Mat<Complex> &M, arma::Mat<Complex> &S,  
                          double thresh) {
                arma::Mat<Complex> X, eigvec;
                arma::Col<double> eigval;
                libgnme::gen_eig_sym<Complex>(dim, M, S, X, eigval, eigvec, thresh);
                arma::Row<double> roweig = eigval.st();
                return std::make_tuple(roweig, eigvec);
          },
          py::arg("dim"),
          py::arg("M"),
          py::arg("S"),
          py::arg("thresh") = 1e-8
    );
    m.def("adjoint_matrix", libgnme::adjoint_matrix<double>);
    m.def("adjoint_matrix", libgnme::adjoint_matrix<Complex>);
}

/*
 * Export eri_ao2mo
 */
void export_eri_ao2mo(py::module &m) {
    using complex = std::complex<double>;
    m.def("eri_ao2mo", libgnme::eri_ao2mo<double, double>);
    m.def("eri_ao2mo", libgnme::eri_ao2mo<Complex, double>);
    m.def("eri_ao2mo", libgnme::eri_ao2mo<Complex, Complex>);
    m.def("eri_ao2mo_split", libgnme::eri_ao2mo<double, double>);
    m.def("eri_ao2mo_split", libgnme::eri_ao2mo<Complex, double>);
    m.def("eri_ao2mo_split", libgnme::eri_ao2mo<Complex, Complex>);
}

}  // namespace pybind11:literals

#endif
