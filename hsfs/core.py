# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:22:19 2017

@author: Adam
"""
from operator import attrgetter
import attr
import numpy as np
from tqdm import trange
from sympy.physics.wigner import clebsch_gordan, wigner_3j
from .drake1999 import quantum_defects
from .numerov import rad_overlap

#CODATA 2014, DOI: 10.1103/RevModPhys.88.035009
c = 299792458.0 ## speed of light in vacuum
h = 6.626070040e-34
hbar = 1.054571800e-34
Ry = 10973731.568508
e = 1.6021766208e-19
m_e = 9.10938356e-31
alpha = 7.2973525664e-3
m_u = 1.660539040e-27
En_h = alpha**2.0 * m_e * c**2.0
a_0 = hbar/ (m_e * c * alpha)
## helium
A_r_helium = 4.002603254130
ionization_helium = 1.9831066637e7
mass_helium = A_r_helium * m_u
mass_helium_core = mass_helium - m_e + ionization_helium * h /c
## reduced electron mass/ m_e
mu_me = mass_helium_core / (mass_helium_core + m_e)
## reduced electron mass / core mass,
mu_M = m_e / (mass_helium_core + m_e)
## Rydberg constant for helium
Ry_M = Ry * mu_me

@attr.s()
class State(object):
    """ attrs class to represent the quantum state |n, L, S, J, MJ >.
    """
    n = attr.ib(convert=int)
    @n.validator
    def check_n(self, attribute, value):
        if not value > 0:
            raise ValueError("n must be a positive integer.")
    L = attr.ib(convert=int)
    @L.validator
    def check_L(self, attribute, value):
        if not value < self.n:
            raise ValueError("L must be an integer smaller than n.")
    S = attr.ib(convert=int)
    @S.validator
    def check_S(self, attribute, value):
        if value not in [0, 1]:
            raise ValueError("S must be an integer 0 or 1.")
        elif self.n == 1 and value != 0:
            raise ValueError("if n == 1, S must be 0.")
    J = attr.ib(convert=int)
    @J.validator
    def check_J(self, attribute, value):
        if self.L == 0 and not value == self.S:
            raise ValueError("If L == 0, J must be equal to S.")
        elif (not self.L - self.S <= value <= self.L + self.S):
            raise ValueError("J must be in range L - S < J < L + S.")
    MJ = attr.ib(convert=int)
    @MJ.validator
    def check_MJ(self, attribute, value):
        if (not -self.J <= value <= self.J):
            raise ValueError("MJ must be in the range or -J to J.")

    def __attrs_post_init__(self):
        self.qd = get_qd(self.n, self.L, self.S, self.J)
        self.n_eff = self.n - self.qd
        self.E0 = energy(self.n, self.n_eff)

def get_qd(n, L, S, J, **kwargs):
    """ calculate the quantum defect.
    """
    max_iterations = kwargs.get('max_iterations', 10)
    precision = kwargs.get('precision', 1e-12)
    if L in quantum_defects[S]:
        if J in quantum_defects[S][L]:
            # quantum defect components
            delta = quantum_defects[S][L][J]
            # iteratively calculate quantum defects
            qd_sub1 = delta[0]
            for i in range(max_iterations):
                m = n - qd_sub1
                qd = delta[0]
                for j, d in enumerate(delta[1:]):
                    qd = qd + d * m**(-2*(j + 1))
                # check convergence
                if abs(qd - qd_sub1) < precision:
                    break
                else:
                    qd_sub1 = qd
        else:
            qd = np.nan
    else:
        qd = 0.0
    return qd

def energy(n, n_eff, Z=1):
    """ the ionization energy (atomic units) with relativistic and finite mass corrections.
        Drake 1999 (doi: 10.1238/Physica.Topical.083a00083), eqn. 21
    """
    # TODO - special case for n=1
    en = -0.5 * (1.0 / n_eff**2.0 - \
                 3.0 * alpha**2.0 / (4.0 * n**4.0) + \
                 mu_M**2.0 * ((1 + (5.0/ 6.0) * (Z * alpha)**2.0) / n**2.0))
    return mu_me * en

class StarkMatrix(object):
    """ class to represent Stark matrix.
    """
    def __init__(self, nmin, nmax, S=1, MJ=0):
        self.nmin = nmin
        self.name = nmax
        self.S = S
        self.MJ = MJ
        if S == 1:
            self.basis = triplet_states(nmin, nmax, MJ)
        self.sort_basis('E0', inplace=True)
        self.num_states = len(self.basis)
        self._h0_matrix = None
        self._stark_matrix = None

    def sort_basis(self, attribute, inplace=False):
        """ sort basis on attribute.
        """
        sorted_basis = sorted(self.basis, key=attrgetter(attribute))
        if inplace:
            self.basis = sorted_basis
        return sorted_basis

    def attr(self, attribute):
        """ list of attributes from basis, e.g., qd.
        """
        return [getattr(el, attribute) for el in self.basis]

    def where(self, attribute, value):
        """ index of where basis.attribute == value.
        """
        arr = self.attr(attribute)
        return [i for i, x in enumerate(arr) if x == value]

    def h0_matrix(self, cache=True):
        """ unperturbed Hamiltonian.
        """
        if self._h0_matrix is None or cache is False:
            self._h0_matrix = np.diag(self.attr('E0'))
        return self._h0_matrix

    def stark_matrix(self, cache=True, **kwargs):
        """ Stark interaction matrix.
        """
        if self._stark_matrix is None or cache is False:
            self._stark_matrix = np.zeros([self.num_states, self.num_states])
            for i in trange(self.num_states, desc="calculate Stark terms", **kwargs):
                for j in range(i + 1, self.num_states):
                    self._stark_matrix[i][j] = stark_int(self.basis[i], self.basis[j])
                    # assume matrix is symmetric
                    self._stark_matrix[j][i] = self._stark_matrix[i][j]
        return self._stark_matrix

    def stark_map(self, field, **kwargs):
        """ calculate the eigenvalues for H_0 + H_S
            field in units of V / m
        """
        num_fields = len(field)
        # initialise output arrays
        eig_val = np.empty((num_fields, self.num_states), dtype=float)
        # loop over field values
        mat_s = self.stark_matrix(**kwargs)
        for i in trange(num_fields, desc="diagonalise Hamiltonian", **kwargs):
            F = field[i] * e * a_0 / En_h
            H_S = F * mat_s / mu_me
            # diagonalise, assuming matrix is Hermitian.
            eig_val[i] = np.linalg.eigh(self.h0_matrix() + H_S)[0]
        return eig_val * En_h

def triplet_states(nmin, nmax, MJ=0):
    """ list triplet states in the range of n_min to n_max for given MJ.
    """
    S = 1
    basis = []
    n_rng = np.arange(nmin, nmax + 1, dtype='int')
    for n in n_rng:
        L_rng = np.arange(0, n, dtype='int')
        for L in L_rng:
            if L == 0:
                J = S
                if J >= abs(MJ):
                    basis.append(State(n, L, S, J, MJ))
            else:
                for J in [L + S, L, L - S]:
                    if J >= abs(MJ):
                        basis.append(State(n, L, S, J, MJ))
    return basis

def ang_overlap(l1, l2, m):
    """ angular overlap <l1, m| cos(theta) |l2, m>.
    """
    delta_l = l2 - l1
    if delta_l == -1 and l1 > abs(m):
        return ((l1**2 - m**2) / ((2*l1 + 1) * (2*l1 - 1)))**0.5
    elif delta_l == +1 and l1 + 1 > abs(m):
        return (((l1 + 1)**2 - m**2) / ((2*l1 + 3) * (2*l1 + 1)))**0.5
    else:
        return 0.0

def stark_int(state_1, state_2):
    """ Stark interaction between two states.
    """
    delta_L = state_1.L - state_2.L
    delta_S = state_1.S - state_2.S
    #delta_J = state_1.J - state_2.J
    delta_MJ = state_1.MJ - state_2.MJ
    if abs(delta_L) == 1 and \
       delta_S == 0 and \
       delta_MJ == 0:
        MS = np.arange(-state_1.S, state_1.S + 1)
        ML = state_1.MJ - MS
        tmp = []
        for m in ML:
            tmp.append(float(clebsch_gordan(state_1.L, state_1.S, state_1.J,
                                            m, state_1.MJ - m, state_1.MJ)) * \
                       float(clebsch_gordan(state_2.L, state_2.S, state_2.J,
                                            m, state_2.MJ - m, state_2.MJ)) * \
                       ang_overlap(state_1.L, state_2.L, m))
        # Stark interaction
        return np.sum(tmp) * rad_overlap(state_1.n_eff, state_1.L, state_2.n_eff, state_2.L)
    else:
        return 0.0

def stark_int_alt(state_1, state_2):
    """ Stark interaction between two states (alternate version)
    """
    delta_L = state_2.L - state_1.L
    delta_S = state_2.S - state_1.S
    #delta_J = state_1.J - state_2.J
    delta_MJ = state_2.MJ - state_1.MJ
    if abs(delta_L) == 1 and \
       delta_S == 0 and \
       delta_MJ == 0:
        MS = np.arange(-state_1.S, state_1.S + 1)
        ML = state_1.MJ - MS
        tmp = []
        for m in ML:
            tmp.append((-1)**(state_1.L - state_1.S + state_1.MJ) * (2*state_1.J + 1)**0.5 \
                       * wigner_3j(state_1.L, state_1.S, state_1.J, m, state_1.MJ - m, -state_1.MJ) * \
                       (-1)**(state_2.L - state_2.S + state_2.MJ) * (2*state_2.J + 1)**0.5 \
                       * wigner_3j(state_2.L, state_2.S, state_2.J, m, state_2.MJ - m, -state_2.MJ) *
                       ang_overlap(state_1.L, state_2.L, m))
        # Stark interaction
        return np.sum(tmp) * rad_overlap(state_1.n_eff, state_1.L, state_2.n_eff, state_2.L)
    else:
        return 0.0
