


#%%
from dedicate_code.data_etl.etl_utils import  gen_xy_mesh, gen_uniform_delta
import fipy as fp
import numpy as np



num_of_cells=10**3
nxy=10**3
delta_space = gen_uniform_delta(num_of_cells, length=2)
mesh, xy =gen_xy_mesh(num_of_cells, delta_space, start=[[-0.75],[-0.5]])


K= 1.0
fpnp = fp.numerix
analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
den = 2*fpnp.pi*K
X = np.linalg.norm([mesh.x.value, mesh.y.value], ord=1, axis=0) #mesh.x.value   mesh.cellCenters[0].value


#%%
analyt_sol[:] = - (fpnp.log(X/1.0) / den)

# %%


analyt_unif_2d = analyt_sol.value.reshape((num_of_cells, num_of_cells))
analyt_unif_2d



# %%
import matplotlib.pyplot as plt
plt.plot(xy[0], analyt_unif_2d[nxy//2])

# %%

plt.plot(xy[1], analyt_unif_2d[:, nxy//2], label="analytical")
# %%
