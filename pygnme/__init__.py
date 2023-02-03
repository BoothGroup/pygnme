"""
pygnme
======
"""

import libpygnme
from libpygnme import wick, slater, utils


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
