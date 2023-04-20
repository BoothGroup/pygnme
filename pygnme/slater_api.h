#ifndef PYGNME_SLATER_API_H_
#define PYGNME_SLATER_API_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

//#include "libgnme/slater/slater_condon.h"
#include "libgnme/slater/slater_rscf.h"
#include "libgnme/slater/slater_uscf.h"

namespace py = pybind11;

//namespace pybind11::literals {

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

//} // namespace:pybind11

#endif
