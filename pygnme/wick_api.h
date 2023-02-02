#ifndef PYGNME_WICK_API_H_
#define PYGNME_WICK_API_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "wick_orbitals.h"
#include "wick_rscf.h"
#include "bitset.h"

namespace py = pybind11;

namespace pybind11::literals {


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
                [](WickRscf &scf, Bitset &bxa, Bitset &bxb, Bitset &bwa, Bitset &bwb) {
                    Tc S = 0.0;
                    Tc V = 0.0;
                    scf.evaluate(bxa, bxb, bwa, bwb, S, V);
                    return std::make_tuple(S, V);
                }
        )
        .def("evaluate", 
                [](WickRscf &scf, arma::umat &xa_hp, arma::umat &xb_hp, arma::umat &wa_hp, arma::umat &wb_hp, Tc &S, Tc &V) {
                    scf.evaluate(xa_hp, xb_hp, wa_hp, wb_hp, S, V);
                    return std::make_tuple(S, V);
                }
        )
        .def("evaluate", 
                [](WickRscf &scf, arma::umat &xa_hp, arma::umat &xb_hp, arma::umat &wa_hp, arma::umat &wb_hp) {
                    Tc S = 0.0;
                    Tc V = 0.0;
                    scf.evaluate(xa_hp, xb_hp, wa_hp, wb_hp, S, V);
                    return std::make_tuple(S, V);
                }
        );
}

}  // namespace pybind11:literals

#endif
