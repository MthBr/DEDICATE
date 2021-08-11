#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))



#%%  Import pakages
from dedicate_code.feature_eng.delta_func_utils import delta_func, delta_radial_func

import numpy as np
import matplotlib.pyplot as plt

#%%  Setups
from dedicate_code.data_etl.etl_utils import  gen_uniform_delta
n_cl = number_of_cells = 16 #1024 #256 # 1024
start=-3.0
leng=6.0
delta_space = gen_uniform_delta(number_of_cells, leng)
epsilon = 1

moment = 3
delta_type = 'pCubic-s2'  #  '3*f-arc'    '4*f'
delta_typ_descript = '3-point smoothed function'


#%%  Dimension 1
from dedicate_code.data_etl.etl_utils import gen_x_mesh, array_from_mesh
dim = 1
mesh =gen_x_mesh(number_of_cells, delta_space, start)
x = array_from_mesh(dim, mesh)


#%% Dimension 1
fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(1, 1, 1)
delta_value =delta_func(mesh, 0.0, epsilon, dim, delta_type)
#delta_value[ delta_value==0 ] = np.nan
ax.plot(x[0], delta_value)
ax.set_title(delta_typ_descript)
ax.set_xlabel(r'$x$') 
ax.set_ylabel(r'$\delta$')
ax.set_xlim(-3.0,3.0)


delta_type2 = 'p2-Cubic'  #  '3*f-arc'    '4*f'
delta_value2 =delta_func(mesh, 0.0, epsilon, dim, delta_type2)
#delta_value2[ delta_value2==0 ] = np.nan
ax.plot(x[0], delta_value2, linestyle='dashed')



# %%
np.testing.assert_array_equal(delta_value, delta_value2)
# %%
