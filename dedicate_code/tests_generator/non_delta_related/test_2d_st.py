#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))

#%%  Import pakages
from dedicate_code.setup_delta_tests import setup_dict
from dedicate_code.tests.test_st_utils import plot_conv_acc, calulate_all_errors, plot_final_funcs_xy

import numpy as np
import matplotlib.pyplot as plt
import fipy as fp

##%%
#import matplotlib.pyplot as plt
#plt.style.use('seaborn-notebook')


#%%  Error testing 
def find_error(mesh, numerc_sol):
        global setup_dict
        K= setup_dict['D']
        fpnp = fp.numerix
        analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
        den = 2*fpnp.pi*K
        X = np.linalg.norm([mesh.x.value, mesh.y.value], ord=2, axis=0)
        analyt_sol[:] =  (fpnp.log(X/1.0) / den)
        error = (fpnp.sum((numerc_sol - analyt_sol)**2)/len(numerc_sol))**(0.5)

        print('**********************')
        print(fpnp.allclose(analyt_sol, numerc_sol, atol=1e-3))
        print('**********************')
        return error.value, analyt_sol.value




#%%  Plot single errors
fig =  plot_final_funcs_xy(300, find_error, setup_dict)

#%%  Generate num for convergence accuracy
nxvec = np.arange(512, 1024, 8)
errors_u, errors_nu_seg = calulate_all_errors(nxvec, find_error, setup_dict)

#%%  Plot

fig = plot_conv_acc(nxvec, errors_u, errors_nu_seg, setup_dict)


#fig = plot_conv_acc(nxvec[0:19], errors_u[0:19], errors_nu_seg[0:19])


# %%
