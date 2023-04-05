"""
pygnme
======

The `pygnme` package is a python interface to the `libgnme` package.
"""

__version__ = "1.0.0"

import libpygnme
from numpy import zeros
from libpygnme import wick, slater, utils

def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x

wick.reference_state = {
    (float): wick.reference_state_double,
    (complex): wick.reference_state_complex,
}

wick.wick_orbitals = {
    (float, float): wick.wick_orbitals_double_double,
    (complex, float): wick.wick_orbitals_complex_double,
    (complex, complex): wick.wick_orbitals_complex_complex,
}


wick.wick_rscf = {
    (float, float, float): wick.wick_rscf_double_double_double,
    (complex, float, float): wick.wick_rscf_complex_double_double,
    (complex, complex, float): wick.wick_rscf_complex_complex_double,
    (complex, complex, complex): wick.wick_rscf_complex_complex_complex,
}


wick.wick_uscf = {
    (float, float, float): wick.wick_uscf_double_double_double,
    (complex, float, float): wick.wick_uscf_complex_double_double,
    (complex, complex, float): wick.wick_uscf_complex_complex_double,
    (complex, complex, complex): wick.wick_uscf_complex_complex_complex,
}


slater.slater_rscf = {
    (float, float, float): slater.slater_rscf_double_double_double,
    (complex, float, float): slater.slater_rscf_complex_double_double,
    (complex, complex, float): slater.slater_rscf_complex_complex_double,
    (complex, complex, complex): slater.slater_rscf_complex_complex_complex,
}


slater.slater_uscf = {
    (float, float, float): slater.slater_uscf_double_double_double,
    (complex, float, float): slater.slater_uscf_complex_double_double,
    (complex, complex, float): slater.slater_uscf_complex_complex_double,
    (complex, complex, complex): slater.slater_uscf_complex_complex_complex,
}
