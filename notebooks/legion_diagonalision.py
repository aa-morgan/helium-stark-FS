
# coding: utf-8

# In[ ]:

from hsfs import Hamiltonian
import numpy as np
import os


# In[ ]:

#Parameters to set
n_min = 2
n_max = 3

Bfield=0.1
Efield_vec=[1.0,0.0,0.0]
Efield_mag=0.0

matrices_dir = './'


# In[ ]:

S=None
print(('n_min={}, n_max={}, S={}').format(n_min, n_max, S))
mat0 = Hamiltonian(n_min=n_min, n_max=n_max, S=S)
print('Number of basis states:', '%d'%mat0.num_states)

s_t_coupling=True
if Efield_vec == [0.0,0.0,1.0]:
    field_orientation = 'para'
elif Efield_vec[2] == 0.0:
    field_orientation = 'perp'

Efield = np.linspace(Efield_mag, Efield_mag, 1) # V /cm
print('E_mag={}, E_vec={}, B={},S-T={}'.format(Efield_mag,Efield_vec,Bfield,s_t_coupling))
sm0 = mat0.stark_map(Efield*1e2,  
                     Efield_vec=Efield_vec,
                     Bfield=Bfield,
                     singlet_triplet_coupling=s_t_coupling,
                     cache_matrices=False,
                     load_matrices=True,
                     save_matrices=False,
                     matrices_dir=matrices_dir,
                     tqdm_disable=False)


# In[ ]:

# Save Stark Map
filename_ham = mat0.filename()
filename_fields = 'E_mag={}_E_vec={}_B={}'.format(Efield_mag, field_orientation, Bfield)
filename_full = filename_ham + '__' + filename_fields
directory = matrices_dir
filepath = os.path.join(directory, filename_full)
print('Filepath={}.npy'.format(filepath))
np.save(filepath, sm0)


# In[ ]:



