#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))

#%%  Import pakages
from dedicate_code.setup_delta_tests import setup_dict

import numpy as np
import matplotlib.pyplot as plt
import fipy as fp

#%%


from dedicate_code.data_etl.etl_utils import array_from_mesh , gen_xy_mesh
from dedicate_code.data_etl.etl_utils import gen_uniform_delta
from dedicate_code.feature_eng.solver4stationary_util import single_solve

num_of_cells = 512

delta_space = gen_uniform_delta(num_of_cells, length=2)
mesh =gen_xy_mesh(num_of_cells, delta_space, start=-1)
xy = array_from_mesh(2, mesh)
setup_dict['grid_spacing'] = delta_space
setup_dict['center_spacing'] = delta_space
final_phi = single_solve(setup_dict, mesh, None)  #False  True


#%%

def find_error(mesh, numerc_sol):
        global setup_dict
        K= setup_dict['D']
        fpnp = fp.numerix
        analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
        den = 2*fpnp.pi*K
        X = np.linalg.norm([mesh.x.value, mesh.y.value], ord=2, axis=0)
        analyt_sol[:] =  (fpnp.log(X/1.0) / den)
        error = (fpnp.sum((numerc_sol - analyt_sol)**2)/len(numerc_sol))**(0.5)  
        #relative error
        rel_err= abs(numerc_sol-analyt_sol)/abs(numerc_sol)

        print(fpnp.allclose(rel_err))
        print('**********************')
        print(fpnp.allclose(analyt_sol, numerc_sol, atol=1e-3))
        print('**********************')
        return error.value, analyt_sol.value


#%%

error_u, analyt_unif = find_error(mesh, final_phi)



# %%

from dedicate_code.data_etl.etl_utils import gen_xy_nnunif_mesh
from dedicate_code.data_etl.etl_utils import gen_nonuniform_segment_delta


delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells, length=2)
mesh =gen_xy_nnunif_mesh(delta_vect, start=-1.0)
xy = array_from_mesh(2, mesh, len(delta_vect))
setup_dict['grid_spacing'] = delta_space
setup_dict['center_spacing']  = delta_space
final_phi = single_solve(setup_dict, mesh, None)  #False  True


#%%

errors_nu_seg, analyt_unif = find_error(mesh, final_phi)
# %%
