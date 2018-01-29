# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:22:19 2017

@author: Adam
"""
from operator import attrgetter
import attr
import numpy as np
from tqdm import trange
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j
from .drake1999 import quantum_defects
from .numerov import rad_overlap
import pandas as pd
import os.path

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
mu_B = e * hbar / (2.0 * m_e)

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
## g-factors
g_L = 1 - m_e / mass_helium_core
g_s = 2.00231930436182

# Other parameters
CACHE_DEFAULT = False

@attr.s()
class State(object):
    """ attrs class to represent the quantum state |n l S J MJ>.
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
        
    def __str__(self):
        """ print quantum numbers like |n l S J MJ >
        """
        return u"\u2758 {} {} {} {} {} \u27E9".format(self.n, self.L, self.S, self.J, self.MJ)
    
    def asdict(self):
        """ quantum numbers as a dictionary.
        """
        return attr.asdict(self)

    def tex(self, show_MJ=True):
        """ Tex string of the form n^{2S + 1}L_{J} (M_J = {MJ})
        """
        L = 'SPDFGHIKLMNOQRTUVWXYZ'[int(self.l%22)]
        tex_str = r'$%d^{%d}'%(self.n, 2*self.S + 1) + L + r'_{%d}'%(self.J)
        if show_MJ:
            tex_str = tex_str + '\,' + r'(M_J = %d)$'%self.MJ
        else:
            tex_str = tex_str + r'$'
        return tex_str

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

class Hamiltonian(object):
    """ The total Hamiltonian matrix.  Each element of the basis set is an
        instance of the class 'State', which represents |n l S J MJ>.
    """
    def __init__(self, n_min, n_max, l_max=None, S=None, MJ=None, MJ_max=None):
        self.n_min = n_min
        self.n_max = n_max
        self.l_max = l_max
        self.S = S
        self.MJ = MJ
        self.MJ_max = MJ_max
        self.basis = basis_states(n_min, n_max, l_max=l_max, S=S, MJ=MJ, MJ_max=MJ_max)
        self.sort_basis('E0', inplace=True)
        self.num_states = len(self.basis)
        self._h0_matrix = None
        self._stark_matrix = None
        self._zeeman_matrix = None
        self._singlet_triplet_coupling_matrix = None
      
    def sort_basis(self, attribute, inplace=False):
        """ Sort basis on attribute.
        """
        sorted_basis = sorted(self.basis, key=attrgetter(attribute))
        if inplace:
            self.basis = sorted_basis
        return sorted_basis

    def attrib(self, attribute):
        """ List of given attribute values from all elements in the basis, e.g., J or E0.
        """
        return [getattr(el, attribute) for el in self.basis]

    def where(self, attribute, value):
        """ Indexes of where basis.attribute == value.
        """
        arr = self.attrib(attribute)
        return [i for i, x in enumerate(arr) if x == value]
    
    def filename(self):
        return  'n=' + str(self.n_min) + '-' + str(self.n_max) + '_' + \
                'l_max=' + str(self.l_max) + '_' + \
                'S=' + str(self.S) + '_' + \
                'MJ=' + str(self.MJ) + '_' + \
                'MJ_max=' + str(self.MJ_max)

    def h0_matrix(self, **kwargs):
        """ Unperturbed Hamiltonian.
        """
        cache = kwargs.get('cache_matrices', CACHE_DEFAULT)
        if self._h0_matrix is None or cache is False:
            self._h0_matrix = np.diag(self.attrib('E0'))
        return self._h0_matrix

    def stark_matrix(self, **kwargs):
        """ Stark interaction matrix.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        cache = kwargs.get('cache_matrices', CACHE_DEFAULT)
        if self._stark_matrix is None or cache is False:
            if kwargs.get('load_matrices', False) and \
               check_matrix('stark', self, **kwargs):
                self._stark_matrix = load_matrix('stark', self, **kwargs)
            else:
                self._stark_matrix = np.zeros([self.num_states, self.num_states])
                for i in trange(self.num_states, desc="calculate Stark terms", **tqdm_kwargs):
                    # off-diagonal elements only
                    for j in range(i + 1, self.num_states):
                        self._stark_matrix[i][j] = stark_int(self.basis[i], self.basis[j], **kwargs)
                        # assume matrix is symmetric
                        self._stark_matrix[j][i] = self._stark_matrix[i][j]
        else:
            print('Using cached Stark matrix') 
        if kwargs.get('save_matrices', False):
            save_matrix(self._stark_matrix, 'stark', self, **kwargs)
        return self._stark_matrix

    def zeeman_matrix(self, **kwargs):
        """ Zeeman interaction matrix.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        cache = kwargs.get('cache_matrices', CACHE_DEFAULT)
        if self._zeeman_matrix is None or cache is False:
            if kwargs.get('load_matrices', False) and \
               check_matrix('zeeman', self, **kwargs):
                self._zeeman_matrix = load_matrix('zeeman', self, **kwargs)
            else:
                self._zeeman_matrix = np.zeros([self.num_states, self.num_states])
                for i in trange(self.num_states, desc="calculate Zeeman terms", **tqdm_kwargs):
                    for j in range(i, self.num_states):
                        self._zeeman_matrix[i][j] = zeeman_int(self.basis[i], self.basis[j], **kwargs)
                        # assume matrix is symmetric
                        if i != j:
                            self._zeeman_matrix[j][i] = self._zeeman_matrix[i][j]
        else:
            print('Using cached Zeeman matrix')
        if kwargs.get('save_matrices', False):
            save_matrix(self._zeeman_matrix, 'zeeman', self, **kwargs)
        return self._zeeman_matrix
    
    def singlet_triplet_coupling_matrix(self, **kwargs):
        """ Singlet-Triplet coupling matrix
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        cache = kwargs.get('cache_matrices', CACHE_DEFAULT)
        if self._singlet_triplet_coupling_matrix is None or cache is False:
            if kwargs.get('load_matrices', False) and \
               check_matrix('singlet-triplet', self, **kwargs):
                self._singlet_triplet_coupling_matrix = load_matrix('singlet-triplet', self, **kwargs)
            else:
                self._singlet_triplet_coupling_matrix = np.zeros([self.num_states, self.num_states])
                for i in trange(self.num_states, desc="calculate singlet-triplet coupling terms", **tqdm_kwargs):
                    for j in range(i, self.num_states):
                        self._singlet_triplet_coupling_matrix[i][j] = singlet_triplet_coupling_int(self.basis[i], self.basis[j], **kwargs)
                        # assume matrix is symmetric
                        if i != j:
                            self._singlet_triplet_coupling_matrix[j][i] = self._singlet_triplet_coupling_matrix[i][j]
        else:
            print('Using cached Singlet-Triplet matrix')
        if kwargs.get('save_matrices', False):
            save_matrix(self._singlet_triplet_coupling_matrix, 'singlet-triplet', self, **kwargs)
        return self._singlet_triplet_coupling_matrix

    def stark_zeeman(self, Efield, Bfield=0.0, **kwargs):
        """ Diagonalise the total Hamiltonian, H_0 + H_S + H_Z, for parallel 
            electric and magnetic fields.
        
            args:
                Efield           dtype: float     units: V / m      

                Bfield=0.0       dtype: float     units: T
            
            kwargs:
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors.

                eig_amp=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_amp=[1, 3, 5].
                                 Requires eig_vec=False.
            
        """
        get_eig_vec = kwargs.get('eig_vec', False)
        eig_elements = kwargs.get('eig_amp', None)
        # magnetic field
        if Bfield != 0.0:
            Bz = mu_B * Bfield / En_h
            mat_z =  self.zeeman_matrix(**kwargs)
            H_Z = Bz * mat_z
        else:
            H_Z = 0.0
        # electric field
        if Efield != 0.0:
            Fz = Efield * e * a_0 / En_h
            mat_s = self.stark_matrix(**kwargs)
            H_S = Fz * mat_s / mu_me
        else:
            H_S = 0.0
        # interaction Hamiltonian
        H_int = H_S + H_Z          
        # diagonalise H_tot, assuming matrix is Hermitian.
        if get_eig_vec:
            # eigenvalues and eigenvectors
            eig_val, eig_vec = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)
            return eig_val * En_h, eig_vec
        elif eig_elements is not None:
            # eigenvalues and partial eigenvector amplitudes
            eig_val, vec = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)
            eig_amp = np.sum(vec[eig_elements]**2.0, axis=0)
            return eig_val, eig_amp
        else:
            # eigenvalues
            eig_val = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)[0]
            return eig_val * En_h

        
    def stark_map(self, Efield, Bfield=0.0, **kwargs):
        """ The eigenvalues of H_0 + H_S + H_Z, for a range of electric fields.
        
            args:
                Efield           dtype: list      units: V / m      

                Bfield=0.0       dtype: float     units: T
            
            kwargs:
                Efield_vec=[0.0,0.0,1.0]    dtype: [float]

                                 specifies the orientation of the electric field.
                                 The quantisation axis is Z-axis
                                 
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors for 
                                 every field value.

                eig_amp=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_amp=[1, 3, 5].
                                 Requires eig_vec=False.

            Nb. A large map with eignvectors can take up a LOT of memory.
        """
        Efield_vec = kwargs.get('Efield_vec', [0.0, 0.0, 1.0])
        if Efield_vec == [0.0,0.0,1.0]:
            field_orientation = 'parallel'
        elif Efield_vec[2] == 0.0:
            field_orientation = 'perpendicular'
        else:
            raise Exception('Arbitrary angles not currently supported. Use either parallel (Efield_vec=[0.0,0.0,1.0]), or perpendicular (Efield_vec[2]=0.0) field.')
        print('Using field orientation: ' + field_orientation)
        
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        get_eig_vec = kwargs.get('eig_vec', False)
        eig_elements = kwargs.get('eig_amp', None)
        num_fields = len(Efield)
        # initialise output arrays
        eig_val = np.empty((num_fields, self.num_states), dtype=float)
        if get_eig_vec:
            eig_vec = np.empty((num_fields, self.num_states, self.num_states), dtype=float)
        elif eig_elements is not None:
            eig_amp = np.empty((num_fields, self.num_states), dtype=float)
        # optional magnetic field
        if Bfield != 0.0:
            Bz = mu_B * Bfield / En_h
            mat_z =  self.zeeman_matrix(**kwargs)
            H_Z = Bz * mat_z
            print('H_Z sum: ', np.sum(H_Z))
        else:
            H_Z = 0.0
        # optional singlet_triplet coupling 
        if kwargs.get('singlet_triplet_coupling', False):
            H_spin = self.singlet_triplet_coupling_matrix(**kwargs)
            print('H_spin sum: ', np.sum(H_spin))
        else:
            H_spin = 0.0
        # loop over electric field values
        mat_s = self.stark_matrix(**kwargs)
        print('mat_s sum: ', np.sum(mat_s))
        for i in trange(num_fields, desc="diagonalise Hamiltonian", **tqdm_kwargs):
            Fz = Efield[i] * e * a_0 / En_h
            H_S = Fz * mat_s / mu_me
            # Full interaction matrix. Unused terms are set to 0.0
            H_int = H_S + H_Z + H_spin
            # diagonalise, assuming matrix is Hermitian.
            if get_eig_vec:
                # eigenvalues and eigenvectors
                eig_val[i], eig_vec[i] = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)
            elif eig_elements is not None:
                # eigenvalues and partial eigenvector amplitudes
                eig_val[i], vec = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)
                eig_amp[i] = np.sum(vec[eig_elements]**2.0, axis=0)            
            else:
                # eigenvalues
                eig_val[i] = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)[0]
        # output
        if get_eig_vec:
            return eig_val * En_h, eig_vec
        elif eig_elements is not None:
            return eig_val * En_h, eig_amp
        else:
            return eig_val * En_h

    def zeeman_map(self, Bfield, Efield=0.0, **kwargs):
        """ The eigenvalues of H_0 + H_S + H_Z, for a range of magnetic fields.
        
            args:
                Bfield           dtype: list      units: T      

                Efield=0.0       dtype: float     units: V / m
            
            kwargs:
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors for 
                                 every field value.

                eig_amp=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_amp=[1, 3, 5].
                                 Requires eig_vec=False.
            
            Nb. A large map with eignvectors can take up a LOT of memory.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        get_eig_vec = kwargs.get('eig_vec', False)
        eig_elements = kwargs.get('eig_amp', None)
        num_fields = len(Bfield)
        # initialise output arrays
        eig_val = np.empty((num_fields, self.num_states), dtype=float)
        if get_eig_vec:
            eig_vec = np.empty((num_fields, self.num_states, self.num_states), dtype=float)
        elif eig_elements is not None:
            eig_amp = np.empty((num_fields, self.num_states), dtype=float)
        # optional electric field
        if Efield != 0.0:
            Fz = Efield * e * a_0 / En_h
            mat_s = self.stark_matrix(**kwargs)
            H_S = Fz * mat_s / mu_me
        else:
            H_S = 0.0
        # optional singlet_triplet coupling 
        if 'singlet_triplet_coupling' in kwargs:
            print('Using Singlet-Triplet coupling')
            H_spin = self.singlet_triplet_coupling_matrix(**kwargs)
        else:
            H_spin = 0.0
        # loop over magnetic field values
        mat_z = self.zeeman_matrix(**kwargs)
        for i in trange(num_fields, desc="diagonalise Hamiltonian", **tqdm_kwargs):
            Bz = mu_B * Bfield[i] / En_h
            H_Z =  Bz * mat_z
            # Full interaction matrix. Unused terms are set to 0.0
            H_int = H_S + H_Z + H_spin
            # diagonalise, assuming matrix is Hermitian.
            if get_eig_vec:
                # eigenvalues and eigenvectors
                eig_val[i], eig_vec[i] = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)
            elif eig_elements is not None:
                # eigenvalues and partial eigenvector amplitudes
                eig_val[i], vec = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)
                eig_amp[i] = np.sum(vec[eig_elements]**2.0, axis=0)            
            else:
                # eigenvalues
                eig_val[i] = np.linalg.eigh(self.h0_matrix(**kwargs) + H_int)[0]
        # output
        if get_eig_vec:
            return eig_val * En_h, eig_vec
        elif eig_elements is not None:
            return eig_val * En_h, eig_amp
        else:
            return eig_val * En_h

def basis_states(n_min, n_max, **kwargs):
    """ Generate the basis set: a list of instances of the attrs class State that 
        satisfy the given ranges of quantum numbers.  By default, all possible 
        states in the range of n_min to n_max are returned.
        
        args:
            n_min             Minimum value of the principal quantum number.

            n_max             Maximum value of the principal quantum number.
        
        kwargs:
            l_max = None      Maximum value of the orbital angular momentum quantum number.
                              If l_max is None 0 < l < n.

            S = None          Value of the total spin quanum number. If S is None S = [0, 1].

            MJ = None         Value of the projection of the total angular momentum
                              quantum number. If MJ is None -J <= MJ <= J.

            MJ_max = None     Maximum of the absolute value of the projection of the
                              total angular momentum quantum number. If MJ_max and MJ
                              are None -J <= MJ <= J.
    """
    l_max = kwargs.get('l_max', None)
    S = kwargs.get('S', None)
    MJ = kwargs.get('MJ', None)
    MJ_max = kwargs.get('MJ_max', None)
    basis = []
    n_rng = np.arange(n_min, n_max + 1, dtype='int')
    # loop over n range
    for n in n_rng:
        if l_max is not None:
            _l_max = min(l_max, n - 1)
        else:
            _l_max = n - 1
        l_rng = np.arange(0, _l_max + 1, dtype='int')
        # loop over l range
        for l in l_rng:
            if S is None:
                # singlet and triplet states
                S_vals = [0, 1]
            else:
                S_vals = [S]
            for _S in S_vals:
                # find all J vals and MJ substates
                if l == 0:
                    J = _S
                    if MJ is None:
                        for _MJ in np.arange(-J, J + 1):
                            if MJ_max is None or abs(_MJ) <= MJ_max:
                                basis.append(State(n, l, _S, J, _MJ))
                    elif -J <= MJ <= J:
                        basis.append(State(n, l, _S, J, MJ))
                elif _S == 0:
                    J = l
                    if MJ is None:
                        for _MJ in np.arange(-J, J + 1):
                            if MJ_max is None or abs(_MJ) <= MJ_max:
                                basis.append(State(n, l, _S, J, _MJ))
                    elif -J <= MJ <= J:
                        basis.append(State(n, l, _S, J, MJ))
                else:
                    for J in [l + _S, l, l - _S]:
                        if MJ is None:
                            for _MJ in np.arange(-J, J + 1):
                                if MJ_max is None or abs(_MJ) <= MJ_max:
                                    basis.append(State(n, l, _S, J, _MJ))
                        elif -J <= MJ <= J:
                            basis.append(State(n, l, _S, J, MJ))
    return basis

def stark_int(state_1, state_2, **kwargs):
    stark_method = kwargs.get('stark_method', '3j')
    if stark_method == '3j':
        return stark_int_3j(state_1, state_2, **kwargs)
    elif stark_method == '6j':
        return stark_int_6j(state_1, state_2, **kwargs)
    else:
        raise Exception('Stark int method not recognised')
    
def stark_int_3j(state_1, state_2, **kwargs):
    """ Stark interaction between two states.
        
        <n' l' S' J' MJ'| H_S |n l S J MJ>.
    """     
    Efield_vec = kwargs.get('Efield_vec', [0.0, 0.0, 1.0])
    if Efield_vec == [0.0,0.0,1.0]: # parallel fields
        field_orientation = 'parallel'
        q_arr   = [0]
        tau_arr = [1.]
    elif Efield_vec[2] == 0.0: # perpendicular fields
        field_orientation = 'perpendicular'
        q_arr   = [1,-1]
        tau_arr = [(1./2)**0.5, (1./2)**0.5]
    
    delta_L = state_1.L - state_2.L
    delta_S = state_1.S - state_2.S
    delta_MJ = state_1.MJ - state_2.MJ  
    # Projection of spin, cannot change
    if abs(delta_L) == 1 and delta_S == 0 and \
     ((field_orientation=='parallel'      and     delta_MJ  == 0) or \
      (field_orientation=='perpendicular' and abs(delta_MJ) == 1)):
        # For accumulating each element in the ML sum
        sum_ML = []
        # Loop through all combination of ML for each state
        for MS_1 in np.arange(-state_1.S, state_1.S + 1):
            for MS_2 in np.arange(-state_2.S, state_2.S + 1):
                delta_MS = MS_1 - MS_2
                # Change in projection of spin:  0, +/- 1
                if ((field_orientation=='parallel'      and     delta_MS  in [0,1]) or \
                    (field_orientation=='perpendicular' and abs(delta_MS) in [0,1])):
                    ML_1 = state_1.MJ - MS_1
                    ML_2 = state_2.MJ - MS_2
                    # For accumulating each element in the angular component, q sum
                    sum_q = []
                    for q, tau in zip(q_arr, tau_arr):
                        sum_q.append(tau * float(wigner_3j(state_2.L, 1, state_1.L, -ML_2, q, ML_1)))
                    # Calculate the angular overlap term using Wigner-3J symbols
                    ang_overlap_stark = ((2*state_2.L+1)*(2*state_1.L+1))**0.5 * \
                                          np.sum(sum_q) * \
                                          wigner_3j(state_2.L, 1, state_1.L, 0, 0, 0)
                    if ang_overlap_stark != 0.0:
                        sum_ML.append(float(clebsch_gordan(state_1.L, state_1.S, state_1.J,
                                      ML_1, state_1.MJ - ML_1, state_1.MJ)) * \
                                      float(clebsch_gordan(state_2.L, state_2.S, state_2.J,
                                      ML_2, state_2.MJ - ML_2, state_2.MJ)) * \
                                      ang_overlap_stark)

        # Stark interaction
        return np.sum(sum_ML) * rad_overlap(state_1.n_eff, state_1.L, state_2.n_eff, state_2.L)
    else:
        return 0.0

def stark_int_6j(state_1, state_2, **kwargs):
    """ Stark interaction between two states.
        
        <n' l' S' J' MJ'| H_S |n l S J MJ>.
    """     
    Efield_vec = kwargs.get('Efield_vec', [0.0, 0.0, 1.0])
    if Efield_vec == [0.0,0.0,1.0]: # parallel fields
        q_arr   = [0]
        tau_arr = [1]
    elif Efield_vec[2] == 0.0: # perpendicular fields
        q_arr   = [1,-1]
        tau_arr = [(1./2)**0.5, -(1./2)**0.5]
    
    delta_L = state_1.L - state_2.L
    delta_S = state_1.S - state_2.S
    delta_MJ = state_1.MJ - state_2.MJ  
    if abs(delta_L) == 1 and delta_S == 0:
        S = state_1.S
        sum_q = []
        for q, tau in zip(q_arr, tau_arr):
            sum_q.append( (-1.)**(int(state_1.J - state_1.MJ)) * \
                        wigner_3j(state_1.J, 1, state_2.J, -state_1.MJ, -q, state_2.MJ) * \
                        (-1.)**(int(state_1.L + S + state_2.J + 1.)) * \
                        np.sqrt((2.*state_1.J+1.) * (2.*state_2.J+1.)) * \
                        wigner_6j(state_1.J, 1., state_2.J, state_2.L, S, state_1.L) * \
                        (-1.)**state_1.L * np.sqrt((2.*state_1.L+1.) * (2.*state_2.L+1.)) * \
                        wigner_3j(state_1.L, 1, state_2.L, 0, 0, 0) * tau)
            
        return np.sum(sum_q) * rad_overlap(state_1.n_eff, state_1.L, state_2.n_eff, state_2.L)
    return 0.0
    
def zeeman_int(state_1, state_2, **kwargs):
    zeeman_method = kwargs.get('zeeman_method', 'hsfs')
    if zeeman_method == 'hsfs':
        return zeeman_int_hsfs(state_1, state_2, **kwargs)
    elif zeeman_method == 'dev':
        return zeeman_int_dev(state_1, state_2, **kwargs)
    else:
        raise Exception('Zeeman int method not recognised')
    
def zeeman_int_hsfs(state_1, state_2, **kwargs):
    """ Zeeman interaction between two states.
    """
    delta_S = state_2.S - state_1.S
    delta_L = state_2.L - state_1.L
    delta_J = state_2.J - state_1.J
    delta_MJ = state_2.MJ - state_1.MJ
    if delta_MJ == 0 and \
       delta_J in [-1, 0, 1] and \
       delta_S == 0 and \
       delta_L == 0:
           L = state_1.L
           MJ = state_1.MJ
           S = state_1.S
           g_L2 = g_L * (((2 * L + 1) * L * (L + 1))/6)**0.5
           g_S2 = g_s * (((2 * S + 1) * S * (S + 1))/6)**0.5
           return (-1)**(1 - MJ) * ((2 * state_1.J + 1) * (2 * state_2.J + 1))**0.5 * \
                  wigner_3j(state_2.J, 1, state_1.J, -MJ, 0, MJ) * 6**0.5 * ( \
                  wigner_6j(L, state_2.J, S, state_1.J, L, 1) * \
                  (-1)**(state_1.J + state_2.J + L + S) * g_L2 + \
                  wigner_6j(state_1.J, state_2.J, 1, S, S, L) * \
                  (-1)**(L + S) * g_S2)
    else:
        return 0.0

def zeeman_int_dev(state_1, state_2, **kwargs):
    """ Zeeman interaction between two states,
    
        <n' l' S' J' MJ'| H_Zeeman |n l S J MJ>.
    """
    delta_S = state_2.S - state_1.S
    delta_L = state_2.L - state_1.L
    delta_MJ = state_2.MJ - state_1.MJ
    if delta_L == 0 and \
       delta_MJ == 0:
        L = state_1.L
        MJ = state_1.MJ
        S = state_1.S
        return  (-1.0)**(state_1.L + state_1.MJ) * \
                ((-1.0)**(state_1.S + state_2.S) + 1. ) * \
                (3.0 * (2*state_2.J + 1) * (2*state_1.J + 1))**0.5 * \
                wigner_3j(state_2.J, 1, state_1.J, -MJ, 0, MJ) * \
                wigner_6j(S, L, state_2.J, state_1.J, 1, S)
    else:
        return 0.0
    
def singlet_triplet_coupling_int(state_1, state_2, **kwargs):
    """ Singlet-Triplet interaction between two states.
        Angular momentum: Understanding spatial aspects in Chemistry and Physics - Richard N. Zare. Eqn. 5.79
    """
    delta_S = state_2.S - state_1.S
    delta_L = state_2.L - state_1.L
    delta_J = state_2.J - state_1.J
    delta_MJ = state_2.MJ - state_1.MJ
    if abs(delta_S) == 1 and \
       delta_J == 0 and \
       delta_MJ == 0 and \
       delta_L == 0:
        L1 = state_1.L
        L2 = 0.0 # inner electron is 1s state
        zeta_1 = rad_overlap(state_1.n, L1, state_1.n, L2, p=-3.0)
        zeta_2 = rad_overlap(state_2.n, L1, state_2.n, L2, p=-3.0)
        return ((zeta_1 * (-1)**(state_1.S + state_2.S + state_1.J + L1 + L2 + 1) * \
                   ((2*state_2.L+1) * (2*state_1.L+1) * (2*state_2.S+1) * (2*state_1.S+1) \
                    * (2*L1+1) * L1 * (L1+1) * (3/2))**0.5 * \
                    wigner_6j(L1, state_1.L, L2, state_2.L, L1, 1)) + \
               (zeta_2 * (-1)**(2*state_1.S + state_1.L + state_2.L + state_1.J + L1 + L2 + 1) * \
                   ((2*state_2.L+1) * (2*state_1.L+1) * (2*state_2.S+1) * (2*state_1.S+1) \
                    * (2*L2+1) * L2 * (L2+1) * (3/2))**0.5 * \
                    wigner_6j(L2, state_1.L, L1, state_2.L, L2, 1))) * \
               wigner_6j(state_1.L, state_1.S, state_1.J, state_2.S, state_2.L, 1) * \
               wigner_6j(0.5, state_1.S, 0.5, state_2.S, 0.5, 1)
    else:
        return 0.0
    
def save_matrix(matrix, matrix_type, ham, **kwargs):
    filename = matrix_type + '_' + ham.filename()
    if matrix_type == 'stark':
        Efield_vec = kwargs.get('Efield_vec', [0.0, 0.0, 1.0])
        if Efield_vec == [0.0,0.0,1.0]:
            filename += '_parallel'
        elif Efield_vec[2] == 0.0:
            filename += '_perpendicular'
            
    save_dir = kwargs.get('matrices_dir', './')
    np.savez_compressed(save_dir+filename, matrix=matrix)
    print('Saved', matrix_type, 'matrix as, ')
    print('\t', save_dir+filename)

def load_matrix(matrix_type, ham, **kwargs):
    filename = matrix_type + '_' + ham.filename() + '.npz'
    if matrix_type == 'stark':
        Efield_vec = kwargs.get('Efield_vec', [0.0, 0.0, 1.0])
        if Efield_vec == [0.0,0.0,1.0]:
            filename += '_parallel'
        elif Efield_vec[2] == 0.0:
            filename += '_perpendicular'
            
    load_dir = kwargs.get('matrices_dir', './')
    mat = np.load(load_dir+filename)
    print('Loaded', matrix_type, 'matrix from, ')
    print('\t', load_dir+filename)
    return mat['matrix']

def check_matrix(matrix_type, ham, **kwargs):
    filename = matrix_type + '_' + ham.filename() + '.npz'
    if matrix_type == 'stark':
        Efield_vec = kwargs.get('Efield_vec', [0.0, 0.0, 1.0])
        if Efield_vec == [0.0,0.0,1.0]:
            filename += '_parallel'
        elif Efield_vec[2] == 0.0:
            filename += '_perpendicular'
    
    load_dir = kwargs.get('matrices_dir', './')
    return os.path.isfile(load_dir+filename) 

def constants_info():
    constant_vals = {
        'speed of light in vacuum, $c$': c,
        'Planks constant, $h$': h,
        'Reduced Planks constant, $\hbar$': hbar,
        'Rydberg constant, $R_{\infty}$': Ry,
        'electron charge, $e$': e,
        'fine structure constant': alpha,
        'atomic mass': m_u,
        'Hatree energy': En_h,
        'Bohr radius, $a_0$': a_0,
        'Bohr magneton, $\mu_B$': mu_B,
        'ionization energy of helium': ionization_helium,
        'mass of helium': mass_helium,
        'mass of helium (a.u.)': A_r_helium,
        'mass of helium core': mass_helium_core,
        'Reduced electron mass / electron mass': mu_me,
        'Reduced electron mass / core mass': mu_M,
        'Rydberg constant for helium': Ry_M
    }
    df = pd.DataFrame(list(constant_vals.items()), columns=['Constant', 'Value'])
    df['Value'] = df['Value'].map('{:.14g}'.format)
    return df