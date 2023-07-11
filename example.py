import numpy as np
import scipy.linalg
from pyscf import gto, scf, mcscf, ao2mo
from pygnme import wick, utils

def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = np.zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x

mol = gto.Mole()
mol.atom = ";".join(["H 0 0 %d" % i for i in range(6)])
mol.basis = "sto-3g"
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

h1e = owndata(mf.get_hcore())
h2e = owndata(ao2mo.restore(1, mf._eri, mol.nao).reshape(mol.nao**2, mol.nao**2))
ovlp = owndata(mf.get_ovlp())
nmo, nocc = mf.mo_occ.size, np.sum(mf.mo_occ > 0)

mc1 = mcscf.CASSCF(mf, 4, 4)
e1, _, ci1, mo1, _ = mc1.kernel()

mc2 = mcscf.CASCI(mf, 2, 2)
e2, _, ci2, mo2, _ = mc2.kernel()

ci = (owndata(ci1), owndata(ci2))
mo = (owndata(mo1), owndata(mo2))
nact = (mc1.ncas, mc2.ncas)
ncore = (mc1.ncore, mc2.ncore)

# Compute coupling terms
nstate = len(ci)
h = np.zeros((nstate, nstate))
s = np.zeros((nstate, nstate))
rdm1 = np.zeros((nstate, nstate, nmo, nmo))
for x in range(nstate):
    for w in range(x, nstate):
        # Setup biorthogonalised orbital pair
        refx = wick.reference_state[float](nmo, nmo, nocc, nact[x], ncore[x], mo[x])
        refw = wick.reference_state[float](nmo, nmo, nocc, nact[w], ncore[w], mo[w])

        # Setup paired orbitals
        orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

        # Setup matrix builder object
        mb = wick.wick_rscf[float, float, float](orbs, mol.energy_nuc())
        # Add one- and two-body contributions
        mb.add_one_body(h1e)
        mb.add_two_body(h2e)

        vx = utils.fci_bitset_list(nocc-ncore[x], nact[x])
        vw = utils.fci_bitset_list(nocc-ncore[w], nact[w])

        # Loop over FCI occupation strings
        for iwa in range(len(vw)):
            for iwb in range(len(vw)):
                for ixa in range(len(vx)):
                    for ixb in range(len(vx)):
                        # Compute S and H contribution for this pair of determinants
                        stmp, htmp = mb.evaluate(vx[ixa], vx[ixb], vw[iwa], vw[iwb])
                        h[x, w] += htmp * ci[w][iwa, iwb] * ci[x][ixa, ixb]
                        s[x, w] += stmp * ci[w][iwa, iwb] * ci[x][ixa, ixb]

                        # Compute RDM1 contribution for this pair of determinants
                        tmpP1 = np.zeros((orbs.m_nmo, orbs.m_nmo))
                        mb.evaluate_rdm1(vx[ixa], vx[ixb], vw[iwa], vw[iwb], stmp, tmpP1)
                        rdm1[x, w] += tmpP1 * ci[w][iwa, iwb] * ci[x][ixa, ixb]

        rdm1[x, w] = np.linalg.multi_dot((mo[w], rdm1[x, w], mo[x].T))

        h[w, x] = h[x, w]
        s[w, x] = s[x, w]
        rdm1[w, x] = rdm1[x, w].T

for x in range(nstate):
    for w in range(nstate):
        print("\n RDM-1 for ⟨Ψ_%d| and |Ψ_%d⟩" % (x, w))
        print(rdm1[x, w])

print(e1,e2)
print("\n Hamiltonian")
print(h)
print("\n Overlap")
print(s)

w, v = scipy.linalg.eigh(h, b=s)
print("\n Recoupled NO-CAS-CI eigenvalues")
print(w)
print("\n Recoupled NO-CAS-CI eigenvectors")
print(v)
