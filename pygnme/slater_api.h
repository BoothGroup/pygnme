#ifndef PYGNME_SLATER_API_H_
#define PYGNME_SLATER_API_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

//#include "libgnme/slater/slater_condon.h"
#include "libgnme/slater/slater_rscf.h"
#include "libgnme/slater/slater_uscf.h"

namespace py = pybind11;

namespace pybind11::literals {


/*
 * Export slater_condon
 * FIXME: this isn't even used?
 */
/*
template<typename Tc, typename Tf, typename Tb>
void export_slater_condon(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("slater_condon_") + typestr;
    using SlaterCondon = libgnme::slater_condon<Tc, Tf, Tb>;
    py::class_<SlaterCondon>(m, pyclass_name.c_str())
        // Constructors:
        .def(py::init<const size_t, const size_t, const size_t, const size_t,
                      const arma::Mat<Tb> &>())
        // Variables:
        .def_readonly("m_nbsf", &SlaterCondon::m_nbsf)
        .def_readonly("m_nmo", &SlaterCondon::m_nmo)
        .def_readonly("m_nalpha", &SlaterCondon::m_nalpha)
        .def_readonly("m_nbeta", &SlaterCondon::m_nbeta)
        .def_readwrite("m_Vc", &SlaterCondon::m_Vec)
        .def_readwrite("m_F", &SlaterCondon::m_F)
        .def_readwrite("m_Fa", &SlaterCondon::m_Fa)
        .def_readwrite("m_Fb", &SlaterCondon::m_Fb)
        .def_readwrite("m_II", &SlaterCondon::m_II)
        .def_readwrite("m_one_body", &SlaterCondon::m_one_body)
        .def_readwrite("m_two_body", &SlaterCondon::m_two_body)
        // Functions:
        .def("add_constant", &SlaterCondon::add_constant)
        .def("add_one_body", &SlaterCondon::add_one_body)
        .def("add_two_body", &SlaterCondon::add_two_body)
        // m_metric is a reference:
        .def("m_metric", [](const SlaterCondon &c) { return c.m_metric; })
        // Evaluation functions need return values, since Tc will always be
        // immutable on the python side.
        .def("evaluate", [](SlaterCondon &sc,
                            arma::umat &xa_hp, arma::umat &xb_hp,
                            arma::umat &wa_hp, arma::umat &wb_hp,
                            Tc &S, Tc &M) {
                sc.evaluate(xa_hp, xb_hp, wa_hp, wb_hp, S, M);
                return std::make_tuple(S, M);
            },
            py::arg("xa_hp"),
            py::arg("xb_hp"),
            py::arg("wa_hp"),
            py::arg("wb_hp"),
            py::arg("S") = 0.0,
            py::arg("M") = 0.0
        )
        .def("evaluate_overlap", [](SlaterCondon &sc,
                                    arma::umat &xa_hp, arma::umat &xb_hp,
                                    arma::umat &wa_hp, arma::umat &wb_hp,
                                    Tc &S) {
                sc.evaluate_overlap(xa_hp, xb_hp, wa_hp, wb_hp, S);
                return S;
            },
            py::arg("xa_hp"),
            py::arg("xb_hp"),
            py::arg("wa_hp"),
            py::arg("wb_hp"),
            py::arg("S") = 0.0
        )
        .def("evaluate_one_body_spin", [](SlaterCondon &scf,
                                          arma::umat &xhp, arma::umat &whp,
                                          Tc &S, Tc &V,
                                          bool alpha) {
                scf.evaluate_one_body_spin(xhp, whp, S, V);
                return std::make_tuple(S, V);
            },
            py::arg("xhp"),
            py::arg("whp"),
            py::arg("S") = 0.0,
            py::arg("V") = 0.0,
            py::arg("alpha") = true
        )
        .def("evaluate_rdm1", [](SlaterCondon &scf,
                                 Bitset &bxa, Bitset &bxb,
                                 Bitset &bwa, Bitset &bwb,
                                 Tc &S,
                                 arma::Mat<Tc> &P) {
                // FIXME slater_condon uses 1rdm, I'll stick with rdm1
                scf.evaluate_1rdm(bxa, bxb, bwa, bwb, S, P);
                return S;
            }
        );
}
*/

/*
 * Export slater_rscf
 */
template<typename Tc, typename Tf, typename Tb>
void export_slater_rscf(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("slater_rscf_") + typestr;
    using SlaterRscf = libgnme::slater_rscf<Tc, Tf, Tb>;
    py::class_<SlaterRscf>(m, pyclass_name.c_str())
        // Constructors:
        .def(py::init<const size_t, const size_t, const size_t, const arma::Mat<Tb> &>())
        .def(py::init<const size_t, const size_t, const size_t, const arma::Mat<Tb> &, double>())
        // Functions:
        .def("add_one_body", &SlaterRscf::add_one_body)
        .def("add_two_body", &SlaterRscf::add_two_body)
        .def("evaluate", &SlaterRscf::evaluate)
        .def("evaluate_overlap", &SlaterRscf::evaluate_overlap);
}

/*
 * Export slater_uscf
 */
template <typename Tc, typename Tf, typename Tb>
void export_slater_uscf(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("slater_uscf_") + typestr;
    using SlaterUscf = libgnme::slater_uscf<Tc, Tf, Tb>;
    py::class_<SlaterUscf>(m, pyclass_name.c_str())
        // Constructors:
        .def(py::init<const size_t, const size_t, const size_t, const size_t, const arma::Mat<Tb> &>())
        .def(py::init<const size_t, const size_t, const size_t, const size_t, const arma::Mat<Tb> &, double>())
        // Functions:
        .def("add_one_body", py::overload_cast<arma::Mat<Tf> &>(&SlaterUscf::add_one_body))
        .def("add_one_body", py::overload_cast<arma::Mat<Tf> &, arma::Mat<Tf> &>(&SlaterUscf::add_one_body))
        .def("add_two_body", &SlaterUscf::add_two_body)
        .def("evaluate", &SlaterUscf::evaluate)
        .def("evaluate_overlap", &SlaterUscf::evaluate_overlap);
}

} // namespace:pybind11

#endif
