
# coding: utf-8

from hsfs import Hamiltonian
import numpy as np
import os

#Parameters to set. Make sure there is no space between '=' sign.
_n_min=4
_n_max=5

_Efield_mag=0.0
_Efield_vec=[1.0,0.0,0.0]
_Bfield=0.1

_matrices_dir='./saved_matrices/'

S=None
print(('n_min={}, n_max={}, S={}').format(_n_min, _n_max, S))
mat0 = Hamiltonian(n_min=_n_min, n_max=_n_max, S=S)
print('Number of basis states:', '%d'%mat0.num_states)

use_spin_orbit=True
if _Efield_vec == [0.0,0.0,1.0]:
    field_orientation = 'para'
elif _Efield_vec[2] == 0.0:
    field_orientation = 'perp'

Efield = np.linspace(_Efield_mag, _Efield_mag, 1) # V /cm
print('E_mag={}, E_vec={}, B={},S-T={}'.format(_Efield_mag,_Efield_vec,_Bfield,use_spin_orbit))
sm0 = mat0.stark_map(Efield*1e2, Bfield=_Bfield, 
                     Efield_vec=_Efield_vec, 
                     spin_orbit_coupling=use_spin_orbit,
                     spin_orbit_constants_filename='spin-orbit-constants-values.npy',
                     remove_spin_orbit_from_h0=use_spin_orbit,
                     overwrite_A_diag = None,
                     overwrite_A_off_diag = None,
                     overwrite_SO_offset = None,
                     cache_matrices=False,
                     load_matrices=True,
                     save_matrices=False,
                     matrices_dir=_matrices_dir,
                     tqdm_disable=False)

# Save Stark Map
filename_ham = mat0.filename()
filename_fields = 'E_mag={}_E_vec={}_B={}'.format(_Efield_mag, field_orientation, _Bfield)
filename_full = 'starkMap_' + filename_ham + '__' + filename_fields
directory = _matrices_dir
filepath = os.path.join(directory, filename_full)
print('Filepath={}.npz'.format(filepath))
np.savez_compressed(filepath, matrix=sm0)
