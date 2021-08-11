#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))



#%%  Import pakages
from dedicate_code.feature_eng.delta_func_utils import delta_func

import numpy as np
import matplotlib.pyplot as plt

#%%  Setups
from dedicate_code.data_etl.etl_utils import  gen_uniform_delta
n_cl = number_of_cells = 128 #256 # 1024
start=-1.0
leng=2.0
delta_space = gen_uniform_delta(number_of_cells, leng)
epsilon = 1/2
# 1/3 #1/2 #1.0 
#delta_space*1.0
# delta_space*number_of_cells

moment = [1,1,2,2,2,2,2]
delta_types = ['1-l', '1-l*', '2-l', '2-cos', '2*-cos', '2-Cubic', '2-LL']
delta_typ_descript = ['2-point hat function', 'smoothed \n 2-point hat function', '4-point hat function', '4-point cosine function', \
    'smoothed \n 4-point cosine function \n Class C^2', '2-Cubic',  '2-LL']
n_dt = len(delta_types)


#%%  Dimension 3
from dedicate_code.data_etl.etl_utils import gen_xyz_mesh, array_from_mesh
dim = 3
mesh =gen_xyz_mesh(number_of_cells, delta_space, start)
xyz = array_from_mesh(dim, mesh)

x = xyz[0]
y = xyz[1]
z = xyz[2]

X, Y , Z = np.meshgrid(xyz[0], xyz[1], xyz[2])

#%%  Deltas



idx = 5
delta = delta_types[idx]




delta_value =delta_func(mesh, 0.0, epsilon, dim, delta)
delta_matrix = delta_value.reshape((n_cl, n_cl,n_cl ))




#%%  Deltas

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1, projection='3d')


scat = ax.scatter(X, Y, Z, c=delta_matrix.flatten(), alpha=0.5)
fig.colorbar(scat, shrink=0.5, aspect=5)




#%%  mayavi
from mayavi import mlab
import numpy as np


mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(delta_matrix),
                            plane_orientation='x_axes',
                            slice_index=20,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(delta_matrix),
                            plane_orientation='y_axes',
                            slice_index=20,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(delta_matrix),
                            plane_orientation='z_axes',
                            slice_index=20,
                        )
mlab.outline()
mlab.show()

#%%  Final


title_txt = f'delta Comparisons \n epsilon:{epsilon:.2f}  n:{number_of_cells}'  #
fig.suptitle(title_txt)
fig.tight_layout()
fig.show()




#%%  Salve img


from dedicate_code.config import reportings_dir

folder_name = 'delta'
report_folder = reportings_dir/folder_name
try:
    report_folder.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f"Folder {report_folder} is already there")
else:
    print(f"Folder {report_folder} was created")

file_name = f'delta_TEST_comparisons_rad-tens_ep{int(epsilon*1000)}_n{number_of_cells}'
fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")
# %%





# %%


#%%  Deltas
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D

idx = 1
delta = delta_types[idx]

fig = plt.figure(figsize=(16, 9))


delta_value =delta_func(mesh, 0.0, epsilon, dim, delta)
delta_matrix = delta_value.reshape((n_cl, n_cl,n_cl ))

ax = fig.add_subplot(1, 1, 1, projection='3d')
xslice = 5
yslice = 35
zslice = 0

# Take slices interpolating to allow for arbitrary values
data_x = scipy.interpolate.interp1d(x, delta_matrix, axis=0)(xslice)
data_y = scipy.interpolate.interp1d(y, delta_matrix, axis=1)(yslice)
data_z = scipy.interpolate.interp1d(z, delta_matrix, axis=2)(zslice)


# Pick color map
cmap = plt.cm.plasma
# Plot X slice
xs, ys, zs = delta_matrix.shape
xplot = ax.plot_surface(xslice, y[:, np.newaxis], z[np.newaxis, :],
                        rstride=1, cstride=1, facecolors=cmap(data_x), shade=False)
# Plot Y slice
yplot = ax.plot_surface(x[:, np.newaxis], yslice, z[np.newaxis, :],
                        rstride=1, cstride=1, facecolors=cmap(data_y), shade=False)
# Plot Z slice
zplot = ax.plot_surface(x[:, np.newaxis], y[np.newaxis, :], np.atleast_2d(zslice),
                        rstride=1, cstride=1, facecolors=cmap(data_z), shade=False)

