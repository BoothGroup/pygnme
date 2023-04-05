#ifndef PYGNME_WICK_API_H_
#define PYGNME_WICK_API_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "libgnme/wick/wick_orbitals.h"
#include "libgnme/wick/wick_rscf.h"
#include "libgnme/wick/wick_uscf.h"
#include "libgnme/utils/bitset.h"

namespace py = pybind11;

namespace pybind11::literals {

/* 
 * Export the reference_state class
 */
template<typename Tc> 
void export_reference_state(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("reference_state_") + typestr;
    using ReferenceState = libgnme::reference_state<Tc>;
    using Bitset = libgnme::bitset;
    py::class_<ReferenceState>(m, pyclass_name.c_str())
        // Constructors:
        .def(py::init<const size_t, const size_t, const size_t, 
                      arma::Mat<Tc>>())
        .def(py::init<const size_t, const size_t, const size_t, 
                      const size_t, const size_t,
                      arma::Mat<Tc>>())
        // Variables:
        .def_readonly("m_nbsf",  &ReferenceState::m_nbsf)
        .def_readonly("m_nmo",   &ReferenceState::m_nmo)
        .def_readonly("m_nelec", &ReferenceState::m_nelec)
        .def_readonly("m_nact",  &ReferenceState::m_nact)
        .def_readonly("m_ncore", &ReferenceState::m_ncore)
        .def_readonly("m_C",     &ReferenceState::m_C)
        .def_readwrite("m_bs",   &ReferenceState::m_bs) 
        .def_readwrite("m_core",  &ReferenceState::m_core);
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
        .def(py::init<libgnme::reference_state<Tc> &, 
                      libgnme::reference_state<Tc> &,
                      arma::Mat<Tb>>())
        .def(py::init<const size_t, const size_t, const size_t,
                      arma::Mat<Tc>, arma::Mat<Tc>, arma::Mat<Tb>>())
        // Variables:
        .def_readonly("m_nbsf", &WickOrbitals::m_nbsf)
        .def_readonly("m_nmo", &WickOrbitals::m_nmo)
        .def_readonly("m_nelec", &WickOrbitals::m_nelec)
        .def_readwrite("m_nz", &WickOrbitals::m_nz)
        .def_readwrite("m_redS", &WickOrbitals::m_redS)
        .def_readwrite("m_M", &WickOrbitals::m_M)
        .def_readwrite("m_fX", &WickOrbitals::m_fX)
        .def_readwrite("m_X", &WickOrbitals::m_X)
        .def_readwrite("m_Y", &WickOrbitals::m_Y)
        .def_readwrite("m_CX", &WickOrbitals::m_CX)
        .def_readwrite("m_XC", &WickOrbitals::m_XC)
        .def_readwrite("m_wxP", &WickOrbitals::m_wxP)
        .def_readwrite("m_R", &WickOrbitals::m_R)
        .def_readwrite("m_Q", &WickOrbitals::m_Q)
        .def_readonly("m_refx", &WickOrbitals::m_refx)
        .def_readonly("m_refw", &WickOrbitals::m_refw)
        // m_metric is a reference:
        .def("m_metric", [](const WickOrbitals &self) { return self.m_metric; });
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
        .def(py::init<libgnme::wick_orbitals<Tc, Tb> &, double>(),py::arg("orb"),py::arg("Vc")=0.0)
        // Functions:
        .def("add_one_body", &WickRscf::add_one_body)
        .def("add_two_body", &WickRscf::add_two_body)
        // Evaluation functions need return values, since Tc will always be
        // immutable on the python side.
        .def("evaluate", [](WickRscf &scf,
                            Bitset &bxa, Bitset &bxb,
                            Bitset &bwa, Bitset &bwb,
                            Tc &S, Tc &V) {
                scf.evaluate(bxa, bxb, bwa, bwb, S, V);
                return std::make_tuple(S, V);
            },
            py::arg("bxa"),
            py::arg("bxb"),
            py::arg("bwa"),
            py::arg("bwb"),
            py::arg("S") = 0.0,
            py::arg("V") = 0.0
        )
        .def("evaluate", [](WickRscf &scf,
                            arma::umat &xa_hp, arma::umat &xb_hp,
                            arma::umat &wa_hp, arma::umat &wb_hp,
                            Tc &S, Tc &M) {
                scf.evaluate(xa_hp, xb_hp, wa_hp, wb_hp, S, M);
                return std::make_tuple(S, M);
            },
            py::arg("xa_hp"),
            py::arg("xb_hp"),
            py::arg("wa_hp"),
            py::arg("wb_hp"),
            py::arg("S") = 0.0,
            py::arg("M") = 0.0
        )
        .def("evaluate_rdm1", [](WickRscf &scf,
                                 Bitset &bxa, Bitset &bxb,
                                 Bitset &bwa, Bitset &bwb,
                                 Tc &S, arma::Mat<Tc> &P1) {
                scf.evaluate_rdm1(bxa, bxb, bwa, bwb, S, P1);
                return S;
            },
            py::arg("bxa"),
            py::arg("bxb"),
            py::arg("bwa"),
            py::arg("bwb"),
            py::arg("S"),
            py::arg("P1")
        )
        .def("evaluate_rdm12", [](WickRscf &scf,
                                 Bitset &bxa, Bitset &bxb,
                                 Bitset &bwa, Bitset &bwb,
                                 Tc &S, 
                                 arma::Mat<Tc> &P1, arma::Mat<Tc> &P2) {
                scf.evaluate_rdm12(bxa, bxb, bwa, bwb, S, P1, P2);
                return S;
            },
            py::arg("bxa"),
            py::arg("bxb"),
            py::arg("bwa"),
            py::arg("bwb"),
            py::arg("S"),
            py::arg("P1"),
            py::arg("P2")
        )
        .def("evaluate_overlap", [](WickRscf &scf,
                                    arma::umat &xa_hp, arma::umat &xb_hp,
                                    arma::umat &wa_hp, arma::umat &wb_hp,
                                    Tc &S) {
                scf.evaluate_overlap(xa_hp, xb_hp, wa_hp, wb_hp, S);
                return S;
            },
            py::arg("xa_hp"),
            py::arg("xb_hp"),
            py::arg("wa_hp"),
            py::arg("wb_hp"),
            py::arg("S")
        )
        .def("evaluate_one_body_spin", [](WickRscf &scf,
                                          arma::umat &xhp, arma::umat &whp,
                                          Tc &S, Tc &V) {
                scf.evaluate_one_body_spin(xhp, whp, S, V);
                return std::make_tuple(S, V);
            },
            py::arg("xhp"),
            py::arg("whp"),
            py::arg("S"),
            py::arg("V")
        );
}


/*
 * Export the wick_uscf class
 */
template<typename Tc, typename Tf, typename Tb>
void export_wick_uscf(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("wick_uscf_") + typestr;
    using WickUscf = libgnme::wick_uscf<Tc, Tf, Tb>;
    using Bitset = libgnme::bitset;
    py::class_<WickUscf>(m, pyclass_name.c_str())
        // Constructors:
        .def(py::init<libgnme::wick_orbitals<Tc, Tb> &, 
                      libgnme::wick_orbitals<Tc, Tb> &, double>(),
             py::arg("orba"),py::arg("orbb"),py::arg("Vc")=0.0)
        // Functions:
        .def("add_one_body", py::overload_cast<arma::Mat<Tf> &>(&WickUscf::add_one_body))
        .def("add_one_body", py::overload_cast<arma::Mat<Tf> &, arma::Mat<Tf> &>(&WickUscf::add_one_body))
        .def("add_two_body", &WickUscf::add_two_body)
        // Evaluation functions need return values, since Tc will always be
        // immutable on the python side.
        .def("evaluate", [](WickUscf &scf,
                            Bitset &bxa, Bitset &bxb,
                            Bitset &bwa, Bitset &bwb,
                            Tc &S, Tc &V) {
                scf.evaluate(bxa, bxb, bwa, bwb, S, V);
                return std::make_tuple(S, V);
            },
            py::arg("bxa"),
            py::arg("bxb"),
            py::arg("bwa"),
            py::arg("bwb"),
            py::arg("S") = 0.0,
            py::arg("V") = 0.0
        )
        .def("evaluate", [](WickUscf &scf,
                            arma::umat &xa_hp, arma::umat &xb_hp,
                            arma::umat &wa_hp, arma::umat &wb_hp,
                            Tc &S, Tc &V) {
                scf.evaluate(xa_hp, xb_hp, wa_hp, wb_hp, S, V);
                return std::make_tuple(S, V);
            },
            py::arg("xa_hp"),
            py::arg("xb_hp"),
            py::arg("wa_hp"),
            py::arg("wb_hp"),
            py::arg("S") = 0.0,
            py::arg("V") = 0.0
        )
        .def("evaluate_rdm1", [](WickUscf &scf,
                                 Bitset &bxa, Bitset &bxb,
                                 Bitset &bwa, Bitset &bwb,
                                 Tc &S, 
                                 arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb) {
                S = 0.0;
                scf.evaluate_rdm1(bxa, bxb, bwa, bwb, S, Pa, Pb);
                return S;
            },
            py::arg("bxa"),
            py::arg("bxb"),
            py::arg("bwa"),
            py::arg("bwb"),
            py::arg("S"),
            py::arg("Pa"),
            py::arg("Pb")
        )
        .def("evaluate_rdm12", [](WickUscf &scf,
                                 Bitset &bxa, Bitset &bxb,
                                 Bitset &bwa, Bitset &bwb,
                                 Tc &S, 
                                 arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b,
                                 arma::Mat<Tc> &P2aa, arma::Mat<Tc> &P2bb,
                                 arma::Mat<Tc> &P2ab) {
                S = 0.0;
                scf.evaluate_rdm12(bxa, bxb, bwa, bwb, S, P1a, P1b, P2aa, P2bb, P2ab);
                return S;
            },
            py::arg("bxa"),
            py::arg("bxb"),
            py::arg("bwa"),
            py::arg("bwb"),
            py::arg("S"),
            py::arg("P1a"),
            py::arg("P1b"),
            py::arg("P2aa"),
            py::arg("P2bb"),
            py::arg("P2ab")
        )
        .def("evaluate_overlap", [](WickUscf &scf,
                                    arma::umat &xa_hp, arma::umat &xb_hp,
                                    arma::umat &wa_hp, arma::umat &wb_hp,
                                    Tc &S) {
                scf.evaluate_overlap(xa_hp, xb_hp, wa_hp, wb_hp, S);
                return S;
            },
            py::arg("xa_hp"),
            py::arg("xb_hp"),
            py::arg("wa_hp"),
            py::arg("wb_hp"),
            py::arg("S") = 0.0
        )
        .def("evaluate_one_body_spin", [](WickUscf &scf,
                                          arma::umat &xhp, arma::umat &whp,
                                          Tc &S, Tc &V,
                                          bool alpha) {
                scf.evaluate_one_body_spin(xhp, whp, S, V, alpha);
                return std::make_tuple(S, V);
            },
            py::arg("xhp"),
            py::arg("whp"),
            py::arg("S") = 0.0,
            py::arg("V") = 0.0,
            py::arg("alpha") = true
        );
}

}  // namespace pybind11:literals

#endif
