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
n_cl = number_of_cells = 1024 #32 #1024 #256 # 1024
start=-3.0
leng=6.0
delta_space = gen_uniform_delta(number_of_cells, leng)
epsilon = 2 #
#epsilon = 1 way as it si shown on paper!!!

# 1/3 #1/2 #1.0 
#delta_space*1.0
# delta_space*number_of_cells

moment = [1,1,2,2,2,2,2]
delta_types = ['l-1-0-1d', 'l-1-1-1d', 'l-1-1-2d', 'l-2-2-1d', 'l-2-2-2d', 'l-1-2-2d', 'l-2-3-1d']
delta_typ_descript = ['l-1-0-1d', 'l-1-1-1d', 'l-1-1-2d', 'l-2-2-1d', 'l-2-2-2d', 'l-1-2-2d', 'l-2-3-1d']
n_dt = len(delta_types)


#%%  Dimension 1
from dedicate_code.data_etl.etl_utils import gen_x_mesh, array_from_mesh
dim = 1
mesh =gen_x_mesh(number_of_cells, delta_space, start)
x = array_from_mesh(dim, mesh)

#%%  Deltas
fig = plt.figure(figsize=(16, 9))

for idx, delta in enumerate(delta_types):
    ax = fig.add_subplot(3, n_dt, idx+1)
    delta_value =delta_func(mesh, 0.0, epsilon, dim, delta)
    ax.set_title(f"{delta_typ_descript[idx]} \n min={np.min(delta_value):.1f} \n max={np.max(delta_value):.1f}")
    #delta_value[ delta_value==0 ] = np.nan
    ax.plot(x[0], delta_value)#, marker='o', linestyle='dashed')
    ax.set_xlabel(r'$x$') 
    ax.set_ylabel(r'$\delta$')
    ax.set_xlim(-3.0,3.0)
    #ax.set_ylim(-0.12, 1.22)


#%%  Dimension 2
from dedicate_code.data_etl.etl_utils import gen_xy_mesh, array_from_mesh
dim = 2
mesh =gen_xy_mesh(number_of_cells, delta_space, start)
xy = array_from_mesh(dim, mesh)
X, Y = np.meshgrid(xy[0], xy[1])
x_labels = [int(j) for j in range(int(start),int(start+leng)+1)]
extend_img=[start,start+leng,start,start+leng]
cmap_deltas = 'hsv'


#%%  Deltas 2D


for idx, delta in enumerate(delta_types):
    ax = fig.add_subplot(3, n_dt, 1*n_dt+idx+1)
    delta_value =delta_func(mesh, 0.0, epsilon, dim, delta)
    delta_matrix = delta_value.reshape((n_cl, n_cl))
    CS = ax.contour(X, Y, delta_matrix, cmap=cmap_deltas, linewidths=0.7, alpha=0.1)
    im = ax.matshow(delta_matrix, extent=extend_img, origin='lower', cmap=cmap_deltas, alpha=0.9)
    ax.set_title(f"min={np.min(delta_value):.2f} \n max={np.max(delta_value):.2f}")
    ax.set_xlabel(r'$x$') 
    ax.set_xlabel(r'$y$') 
    fig.colorbar(im) #, cax=cbar_ax



#%%  Deltas 2D  radial

for idx, delta in enumerate(delta_types):
    ax = fig.add_subplot(3, n_dt, 2*n_dt+idx+1)
    delta_value =delta_radial_func(mesh, 0.0, epsilon, dim, delta)
    delta_matrix = delta_value.reshape((n_cl, n_cl))
    CS = ax.contour(X, Y, delta_matrix, cmap=cmap_deltas, linewidths=0.7, alpha=0.1)
    im = ax.matshow(delta_matrix, extent=extend_img, origin='lower', cmap=cmap_deltas, alpha=0.9)
    ax.set_title(f"min={np.min(delta_value):.2f} \n max={np.max(delta_value):.2f}")
    ax.set_xlabel(r'$x$') 
    ax.set_xlabel(r'$y$') 
    fig.colorbar(im) #, cax=cbar_ax




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

file_name = f'new_delta_comparisons_rad-tens_ep{int(epsilon*1000)}_n{number_of_cells}'
fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")
# %%

# %%

# %%
