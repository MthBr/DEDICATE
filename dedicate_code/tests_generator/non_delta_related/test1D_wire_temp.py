# -*- coding: utf-8 -*-
"""
Version 1, Exploration
Read setup, plot fileds
Read and plot single (1) solution
@author: enzo
"""
#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))

#%%  Import pakages
from dedicate_code.config import data_dir, reportings_dir
from dedicate_code.setup_testSet import setup_dict
from dedicate_code.data_etl.etl_utils import gen_nonuniform_segment_delta, gen_nonuniform_chebyshev_delta, gen_x_nnunif_mesh, gen_x_mesh, gen_uniform_delta
from dedicate_code.feature_eng.solver_util import single_movie_solve, de_f_time

import numpy as np
import matplotlib.pyplot as plt
import fipy as fp

#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

#%% Set style
import matplotlib as mpl
axtickfsize = 16
labelfsize = 20
legfsize = labelfsize - 2
txtfsize = labelfsize - 2
lwidth = 3
markersize = 10
markeredgewidth = 0.1
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = labelfsize
mpl.rcParams['xtick.labelsize'] = axtickfsize
mpl.rcParams['ytick.labelsize'] = axtickfsize
mpl.rcParams['font.size'] = txtfsize
mpl.rcParams['legend.fontsize'] = legfsize
mpl.rcParams['lines.linewidth'] = lwidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = markeredgewidth


#%%  Dataset paths
filed_data = raw_data = data_dir / 'intermediate'
single_run_data = data_dir / 'intermediate'


#%%  set fields file name and soltion file name
#No filed for this example, just set up mesh
number_of_cells=setup_dict['number_of_cells']
delta_vect, delta_space = gen_nonuniform_segment_delta(number_of_cells)
mesh, x =gen_x_nnunif_mesh(delta_vect)

#setup_dict['grid_spacing'] = dxyz

#%%  choose solution i.e. (+) filed number
#No selection for this example
timeDuration = 0.1


#%%  Read fileds and solution
#Only generate single - determinisctic solution solution
setup_dict['grid_spacing'] = delta_space
three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
assert(len(three_time_phi[1][-1])==setup_dict['number_of_cells'])

##%%  Plot selected field
#no plotting needed

#%%  Error testing 


def find_error(mesh, numerc_sol, t):
    fpnp = fp.numerix
    analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
    n = fpnp.array(range(1, 18+1, 2))
    c1=400/(fpnp.pi**3*n**3)
    c3=fpnp.exp(- (n**2) * (fpnp.pi**2)*0.003*t)
    X = mesh.x #mesh.cellCenters[0].value
    analyt_sol[:] = c1[0]* fpnp.sin(X*fpnp.pi)*c3[0] +\
                    c1[1]* fpnp.sin(n[1]*X*fpnp.pi)*c3[1] +\
                    c1[2]* fpnp.sin(n[2]*X*fpnp.pi)*c3[2] +\
                    c1[3]* fpnp.sin(n[3]*X*fpnp.pi)*c3[3] +\
                    c1[4]* fpnp.sin(n[4]*X*fpnp.pi)*c3[4] +\
                    c1[5]* fpnp.sin(n[5]*X*fpnp.pi)*c3[5] +\
                    c1[6]* fpnp.sin(n[6]*X*fpnp.pi)*c3[6] +\
                    c1[7]* fpnp.sin(n[7]*X*fpnp.pi)*c3[7] +\
                    c1[8]* fpnp.sin(n[8]*X*fpnp.pi)*c3[8]

    error = (fpnp.sum((numerc_sol - analyt_sol)**2)/len(numerc_sol))**(0.5)
    return error.value, analyt_sol.value

err, anlyt = find_error(mesh, three_time_phi[1][-1], timeDuration) #0.000001 10**-3
err


#%%  Error testing 

err, anlyt = find_error(mesh, three_time_phi[1][-2], timeDuration/2) #0.000001 10**-3
err

#%%  Plot selected solutions
plt.plot(x[0], anlyt)
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.show()

print(max(anlyt))


#%%  Plot selected solutions
plt.plot(x[0], three_time_phi[1][-1])
plt.xlim(0,1)
#plt.ylim(0,1)
plt.show()



#%%  Generate num for convergence accuracy

nxvec = np.arange(126, 150)
errors_u = np.zeros(len(nxvec))
errors_nu_seg = np.zeros(len(nxvec))
errors_nu_chb = np.zeros(len(nxvec))
for indx, num_of_cells in enumerate(nxvec):
    delta_space = gen_uniform_delta(num_of_cells)
    mesh, x =gen_x_mesh(num_of_cells, delta_space)
    setup_dict['grid_spacing'] = delta_space
    three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
    errors_u[indx], _ = find_error(mesh, three_time_phi[1][-1], timeDuration)
    
    delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells)
    mesh, x =gen_x_nnunif_mesh(delta_vect)
    setup_dict['grid_spacing'] = delta_space
    three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
    errors_nu_seg[indx], _ = find_error(mesh, three_time_phi[1][-1], timeDuration)
    
    delta_vect, delta_space = gen_nonuniform_chebyshev_delta(num_of_cells)
    mesh, x =gen_x_nnunif_mesh(delta_vect)
    setup_dict['grid_spacing'] = delta_space
    three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
    errors_nu_chb[indx], _ = find_error(mesh, three_time_phi[1][-1], timeDuration)

    
#%%  Show convergence accuracy
log_nxvec = np.log(nxvec)
log_errors_u = np.log(errors_u)
log_errors_nu_seg = np.log(errors_nu_seg)
log_errors_nu_chb = np.log(errors_nu_chb)
fit_u = np.polyfit(log_nxvec, log_errors_u, 1)
fit_nu_seg = np.polyfit(log_nxvec, log_errors_nu_seg, 1)
fit_nu_chb = np.polyfit(log_nxvec, log_errors_nu_chb, 1)

fig, axs = plt.subplots(1, 3, figsize=(16, 9))#
#1920x1080 (or 1080p)  figsize=(19.20,10.80)  (16, 9)   

formula=setup_dict['eq_formula']
specifc=setup_dict['text_long']
title_txt = f'{formula} \n {specifc}; T={timeDuration}'
fig.suptitle(title_txt)


ax = axs[0]
ax.plot(log_nxvec, log_errors_u, label="simulation errors")
ax.plot(log_nxvec, np.poly1d(fit_u)(log_nxvec), linestyle="--",
        label="Line: {m:1.2f}x+{b:1.2f}".format(m=fit_u[0], b=fit_u[1]))
ax.legend(loc="best")
ax.set_xlabel("log(nx)")
ax.set_ylabel("log(error)")
ax.set_title("Uniform mesh")

ax = axs[1]
ax.plot(log_nxvec, log_errors_nu_seg, label="simulation errors")
ax.plot(log_nxvec, np.poly1d(fit_nu_seg)(log_nxvec), linestyle="--",
        label="Line: {m:1.2f}x+{b:1.2f}".format(m=fit_nu_seg[0], b=fit_nu_seg[1]))
ax.legend(loc="best")
ax.set_xlabel("log(nx)")
ax.set_ylabel("log(error)")
ax.set_title("Non-uniform \n (segments) mesh")

ax = axs[2]
ax.plot(log_nxvec, log_errors_nu_chb, label="simulation errors")
ax.plot(log_nxvec, np.poly1d(fit_nu_chb)(log_nxvec), linestyle="--",
        label="Line: {m:1.2f}x+{b:1.2f}".format(m=fit_nu_chb[0], b=fit_nu_chb[1]))
ax.legend(loc="best")
ax.set_xlabel("log(nx)")
ax.set_ylabel("log(error)")
ax.set_title("Non-uniform \n (Chebyshev) mesh")


fig.tight_layout()

#%% SAVE
file_name = setup_dict['compactxt']

# for pubblication: Nature: Postscript, Vector EPS or PDF format
#maximum 300 p.p.i.; and min 300dpi
fig.savefig(reportings_dir/(file_name+'.pdf'), bbox_inches="tight")
fig.savefig(reportings_dir/(file_name+'.jpg'), bbox_inches="tight")
fig.savefig(reportings_dir/(file_name+'.png'), bbox_inches="tight")
fig.savefig(reportings_dir/(file_name+'.eps'), format='eps', dpi=300)
fig.savefig(reportings_dir/(file_name+'.svg'), format='svg', dpi=300)

plt.show()
























# %%
x =  np.linspace(0, 1, 15)
def cauchy(x):
    return (1 + x**2)**-1


y = cauchy(x)

plt.plot(y, x, 'o')

# %%

n=30
x =  np.linspace(0, 1, n)
i = np.arange(n, dtype=np.float64)
nodes = np.cos((2*(i+1)-1)/(4*n)*np.pi)



plt.plot(nodes, x, 'o')


# %%

# %%
def dxvec_nonuniform_exp(nx):
    reduce_factor = 0.96
    dx0 = 1/(sum([reduce_factor**i for i in range(nx)]))
    dxvec = np.array([dx0*reduce_factor*np.arcsin((2*(i))/(4*nx)*np.pi) for i in range(nx)])
    return dxvec

n=30
dxvec = dxvec_nonuniform_exp(n)
x =  np.linspace(0, 1, n)


mesh = fp.Grid1D(dx=dxvec)


y = mesh.cellCenters[0].value



# %%
plt.plot(y, x, 'o')
# %%
def dxvec_nonuniform_segments(nx):
    L1, L2, L3 = 0.4, 0.2, 0.4
    n1, n3 = int(0.15*nx), int(0.1*nx)
    n2 = nx - n1 - n3
    dx1, dx2, dx3 = L1/n1, L2/n2, L3/n3
    dxvec = [dx1]*n1 + [dx2]*n2 + [dx3]*n3
    return dxvec


n=30
dxvec = dxvec_nonuniform_segments(n)
x =  np.linspace(0, 1, n)


mesh = fp.Grid1D(dx=dxvec)


y = mesh.cellCenters[0].value



# %%
plt.plot(y, x, 'o')
# %%


dxvec = np.arange(0, 1., 0.1)


mesh = fp.Grid1D(dx=dxvec)
y = mesh.cellCenters[0].value
x =  np.linspace(0, 1, len(y))
# %%
plt.plot(y, x, 'o')


# %%
plt.plot(y, x, 'o')
# %%
def gen_nonuniform_chebyshev_delta(number_of_cells,xmin=0.0, xmax=1.0):
    import numpy as np
    # This function calculates the n Chebyshev points
    number_of_cells=number_of_cells+1 # one lost becaouse of diff
    ns = np.arange(1,number_of_cells+1)
    x = np.cos((2*ns-1)*np.pi/(2*number_of_cells))
    inverse_verc = (xmin+xmax)/2 + (xmax-xmin)*x/2
    dxvec = inverse_verc[:-1]-inverse_verc[1:]
    delta_space=min(dxvec)
    delta_space_max=max(dxvec)
    return dxvec, delta_space

a, b = gen_nonuniform_chebyshev_delta(136)
# %%
