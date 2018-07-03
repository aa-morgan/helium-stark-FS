from .numerov import radial_overlap
import numpy as np
import os.path
from tqdm import trange
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

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

class interaction_matrix:
    """
    """
    def __init__(self, matrix_type, basis, **kwargs):
        self.type = matrix_type.lower()
        self.basis = basis
        self.num_states = len(self.basis.states)
        self.matrix = None
        self.populate_interaction_matrix(**kwargs)
    
    def populate_interaction_matrix(self, **kwargs):
        """ Populate interaction matrix.
        """
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        cache = kwargs.get('cache_matrices', True)
        if self.matrix is None or cache is False:
            if kwargs.get('load_matrices', False) and \
               check_matrix(self.type, self, **kwargs):
                self.matrix = load_matrix(self.type, self, **kwargs)
            else:
                self.matrix = np.zeros([self.num_states, self.num_states])
                for i in trange(self.num_states, desc='Calculating '+self.type+' terms', **tqdm_kwargs):
                    # off-diagonal elements only
                    for j in range(i, self.num_states):
                        self.matrix[i][j] = self.interaction_term(self.basis.states[i], self.basis.states[j], **kwargs)
                        # assume matrix is symmetric
                        self.matrix[j][i] = self.matrix[i][j]
                if kwargs.get('save_matrices', False):
                    save_matrix(self, **kwargs)  
        else:
            print("Using cached '{}' matrix".format(self.type))
            
    def interaction_term(self, state_1, state_2, **kwargs):
        """ Calculate interaction term
        """
        if self.type == 'stark':
            return self.stark_interaction(state_1, state_2, **kwargs)
        elif self.type == 'zeeman':
            return self.zeeman_interaction(state_1, state_2, **kwargs)
        else:
            raise Exception("Interaction term '{}' is not recognised!".format(self.type))
            
    def stark_interaction(self, state_1, state_2, **kwargs):
        stark_method = kwargs.get('stark_method', '3j')
        if stark_method.lower() == '3j':
            return self.stark_interaction_Wigner_3j(state_1, state_2, **kwargs)
        elif stark_method.lower() == '6j':
            return self.stark_interaction_Wigner_6j(state_1, state_2, **kwargs)
        else:
            raise Exception("Stark interaction '{}' method not recognised!".format(stark_method))

    def stark_interaction_Wigner_3j(self, state_1, state_2, **kwargs):
        """ Stark interaction between two states.

            <n' l' S' J' MJ'| H_S |n l S J MJ>.
        """     
        field_angle = kwargs.get('field_angle', 0.0)
        if not np.mod(field_angle, 180.0) == 90.0: # parallel fields
            field_orientation = 'parallel'
        elif not np.mod(field_angle, 180.0) == 0.0: # perpendicular fields
            field_orientation = 'perpendicular'
        else:
            raise Exception('Arbitrary angles not yet supported!')

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
                    if ((field_orientation=='parallel'      and abs(delta_MS) in [0]) or \
                        (field_orientation=='perpendicular' and abs(delta_MS) in [0,1])):
                        ML_1 = state_1.MJ - MS_1
                        ML_2 = state_2.MJ - MS_2
                        if (abs(ML_1) <= state_1.L) and (abs(ML_2) <= state_2.L):
                            _angular_overlap = angular_overlap(state_1.L, state_2.L, ML_1, ML_2, **kwargs)
                            if _angular_overlap != 0.0:
                                sum_ML.append(float(clebsch_gordan(state_1.L, state_1.S, state_1.J,
                                              ML_1, state_1.MJ - ML_1, state_1.MJ)) * \
                                              float(clebsch_gordan(state_2.L, state_2.S, state_2.J,
                                              ML_2, state_2.MJ - ML_2, state_2.MJ)) * \
                                              _angular_overlap)

            # Stark interaction
            return np.sum(sum_ML) * radial_overlap(state_1.n_eff, state_1.L, state_2.n_eff, state_2.L)
        else:
            return 0.0
        
    def stark_interaction_Wigner_6j(self, state_1, state_2, **kwargs):
        """ Stark interaction between two states.

            <n' l' S' J' MJ'| H_S |n l S J MJ>.
        """     
        field_angle = kwargs.get('field angle', 0.0)
        if not np.mod(field_angle, 180.0) == 90.0: # parallel fields
            q_arr   = [0]
            tau_arr = [1.]
        elif not np.mod(field_angle, 180.0) == 0.0: # perpendicular fields
            q_arr   = [1,-1]
            tau_arr = [(1./2)**0.5, -(1./2)**0.5]
        else:
            raise Exception('Arbitrary angles not yet supported!')

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

            return np.sum(sum_q) * radial_overlap(state_1.n_eff, state_1.L, state_2.n_eff, state_2.L)
        return 0.0   

    def zeeman_interaction(self, state_1, state_2, **kwargs):
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
    
    def save_matrix(self, **kwargs):
        filename = self.type + '_' + self.filename()
        if self.type == 'stark':
            field_angle = kwargs.get('field_angle', 0.0)
            filename += '_angle_{}'.format(field_angle)

        save_dir = kwargs.get('matrices_dir', './')
        np.savez_compressed(save_dir+filename, matrix=self.matrix)
        print("Saved '{}' matrix from, ".format(self.type))
        print('\t', save_dir+filename)

    def load_matrix(self, **kwargs):
        filename = self.type + '_' + self.filename()
        if self.type == 'stark':
            field_angle = kwargs.get('field_angle', 0.0)
            filename += '_angle_{}'.format(field_angle)
        filename += '.npz'

        load_dir = kwargs.get('matrices_dir', './')
        mat = np.load(load_dir+filename)
        print("Loaded '{}' matrix from, ".format(self.type))
        print('\t', load_dir+filename)
        return mat['matrix']

    def check_matrix(self, **kwargs):
        filename = self.type + '_' + self.filename()
        if self.type == 'stark':
            field_angle = kwargs.get('field_angle', 0.0)
            filename += '_angle_{}'.format(field_angle)
        filename += '.npz'

        load_dir = kwargs.get('matrices_dir', './')
        return os.path.isfile(load_dir+filename) 
    
    def filename(self):
        return  'n=' + str(self.basis.params.n_min) + '-' + str(self.basis.params.n_max) + '_' + \
                'L_max=' + str(self.basis.params.L_max) + '_' + \
                'S=' + str(self.basis.params.S) + '_' + \
                'ML=' + str(self.basis.params.ML) + '_' + \
                'ML_max=' + str(self.basis.params.ML_max)
                
def angular_overlap(L_1, L_2, M_1, M_2, **kwargs):
    angular_overlap_method = kwargs.get('angular_overlap_method', 'analytical')
    if angular_overlap_method == 'analytical':
        return angular_overlap_analytical(L_1, L_2, M_1, M_2, **kwargs)
    elif angular_overlap_method == 'wigner':
        return angular_overlap_wigner(L_1, L_2, M_1, M_2, **kwargs)
    else:
        raise Exception("Angular overlap method '{}' not recognised!".format(angular_overlap_method))
         
def angular_overlap_wigner(L_1, L_2, M_1, M_2, **kwargs):
    field_angle = kwargs.get('field_angle', 0.0)
    if not np.mod(field_angle, 180.0) == 90.0: # parallel fields
        q_arr   = [0]
        tau_arr = [1.]
    elif not np.mod(field_angle, 180.0) == 0.0: # perpendicular fields
        q_arr   = [1,-1]
        tau_arr = [(1./2)**0.5, (1./2)**0.5]
    else:
        raise Exception('Arbitrary angles not yet supported!')
            
    # For accumulating each element in the angular component, q sum
    sum_q = []
    for q, tau in zip(q_arr, tau_arr):
        sum_q.append(tau * float(wigner_3j(L_2, 1, L_1, -M_2, q, M_1)))
    # Calculate the angular overlap term using Wigner-3J symbols
    _angular_overlap = ((2*L_2+1)*(2*L_1+1))**0.5 * \
                          np.sum(sum_q) * \
                          wigner_3j(L_2, 1, L_1, 0, 0, 0)
    return _angular_overlap
                
def angular_overlap_analytical(L_1, L_2, M_1, M_2, **kwargs):
    """ Angular overlap <l1, m| cos(theta) |l2, m>.
        For Stark interaction
    """
    dL = L_2 - L_1
    dM = M_2 - M_1
    L, M = int(L_1), int(M_1)
    field_angle = kwargs.get('field_angle', 0.0)
    frac_para = np.cos(field_angle*(np.pi/180))**2
    frac_perp = np.sin(field_angle*(np.pi/180))**2
    dM_allow = kwargs.get('dM_allow', [0])
    overlap = 0.0
    if not np.mod(field_angle, 180.0) == 90.0:
        if (dM == 0) and (dM in dM_allow):
            if dL == +1:
                overlap += frac_para * (+(((L+1)**2-M**2)/((2*L+3)*(2*L+1)))**0.5)
            elif dL == -1:
                overlap += frac_para * (+((L**2-M**2)/((2*L+1)*(2*L-1)))**0.5)
        elif (dM == +1) and (dM in dM_allow):
            if dL == +1:
                overlap += frac_para * (-((L+M+2)*(L+M+1)/(2*(2*L+3)*(2*L+1)))**0.5)
            elif dL == -1:
                overlap += frac_para * (+((L-M)*(L-M-1)/(2*(2*L+1)*(2*L-1)))**0.5)
        elif (dM == -1) and (dM in dM_allow):
            if dL == +1:
                overlap += frac_para * (+((L-M+2)*(L-M+1)/(2*(2*L+3)*(2*L+1)))**0.5)
            elif dL == -1:
                overlap += frac_para * (-((L+M)*(L+M-1)/(2*(2*L+1)*(2*L-1)))**0.5)

    if not np.mod(field_angle, 180.0) == 0.0:
        if dM == +1:
            if dL == +1:
                overlap += frac_perp * (+(0.5*(-1)**(M-2*L))  * (((L+M+1)*(L+M+2))/((2*L+1)*(2*L+3)))**0.5)
            elif dL == -1:
                overlap += frac_perp * (-(0.5*(-1)**(-M+2*L)) * (((L-M-1)*(L-M))  /((2*L-1)*(2*L+1)))**0.5)
        elif dM == -1:
            if dL == +1:
                overlap += frac_perp * (+(0.5*(-1)**(M-2*L))  * (((L-M+1)*(L-M+2))/((2*L+1)*(2*L+3)))**0.5)
            elif dL == -1:
                overlap += frac_perp * (-(0.5*(-1)**(-M+2*L)) * (((L+M-1)*(L+M))  /((2*L-1)*(2*L+1)))**0.5)
    return overlap 