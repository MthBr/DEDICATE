

#%%
from dedicate_code.setup_testSet import setup_dict
from dedicate_code.data_etl.etl_utils import gen_xyz_mesh, array_from_mesh
from dedicate_code.data_etl.etl_utils import gen_uniform_delta
from dedicate_code.feature_eng.solver4stationary_util import solve, single_solve


num_of_cells = 150


delta_space = gen_uniform_delta(num_of_cells, length=2)
#%%
mesh =gen_xyz_mesh(num_of_cells, delta_space, start=-1)
xyz = array_from_mesh(2, mesh)



#%%
setup_dict['grid_spacing'] = delta_space
setup_dict['center_spacing'] = delta_space
final_phi = single_solve(setup_dict, mesh, None)  #False  True


# %%











# %%

from fipy import Grid2D, Grid3D, numerix

mesh = Grid3D(nx = 3, ny = 2, nz = 1, dx = 0.5, dy = 2., dz = 4.)

print(numerix.allequal((18, 19, 20), numerix.nonzero(mesh.facesTop)[0])) 

# %%

ignore = mesh.facesTop.value 

mesh = Grid2D(nx = 3, ny = 2, dx = 0.5, dy = 2.)
# %%
print(numerix.allequal((6, 7, 8), numerix.nonzero(mesh.facesTop)[0])) 
True

ignore = mesh.facesTop.value 
