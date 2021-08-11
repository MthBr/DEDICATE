#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))

#%%  Import pakages
from dedicate_code.setup_delta_tests import setup_dict

import numpy as np
import matplotlib.pyplot as plt
import fipy as fp


from dedicate_code.data_etl.etl_utils import array_from_mesh , gen_xy_mesh
from dedicate_code.data_etl.etl_utils import gen_uniform_delta
from dedicate_code.feature_eng.solver4stationary_util import single_solve

#%%
delta_types = ['l-1-0-1d', 'l-1-1-1d', 'l-1-1-2d', 'l-2-2-1d', 'l-2-2-2d', 'l-1-2-2d', 'l-2-3-1d']
#['l-1-1-2d', 'l-1-2-2d', 'l-2-2-2d', 'cos-2-2d', 'p4h-s2', 'p*2-s15', 'pCubic-s2']

#%%

K= setup_dict['D']
fpnp = fp.numerix
den = 2*fpnp.pi*K
tensor_product = True


nxvec = 2 ** np.arange(7,10)  #7-12 coarsest mesh division
num_levels= len(nxvec)

n_plots = min(9, len(delta_types)) #number_of_subplots
Cols = int(n_plots**0.5)
# Compute Rows required
Rows = n_plots // Cols 
Rows += n_plots % Cols

fig, axs_m = plt.subplots(Rows, Cols, figsize=(16, 9)) #, sharex=True, sharey=True
axs = axs_m.flatten()

rates = {}
rates_rel = {}

for idx, delta_tp  in enumerate(delta_types):
    
    errors = np.zeros(num_levels)# error measure(s): E[level][error_type]
    rel_errors = np.zeros(num_levels)
    h_deltas = np.zeros(num_levels) # discretization parameter: h[level]

    for i in range(num_levels):
        num_of_cells = nxvec[i]
        delta_space = gen_uniform_delta(num_of_cells, length=2)
        mesh =gen_xy_mesh(num_of_cells, delta_space, start=-1)
        xy = array_from_mesh(2, mesh)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing'] = delta_space
        setup_dict['delta_type'] = delta_tp
        setup_dict['epsilon_delta'] = 1  # 
        setup_dict['tensor_product'] = tensor_product


        h_deltas[i]=delta_space

        numerc_sol = single_solve(setup_dict, mesh, None)  #False  True
        analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
        X = np.linalg.norm([mesh.x.value, mesh.y.value], ord=2, axis=0)
        l0=2-1
        analyt_sol[:] =  (fpnp.log(X/l0) / den)

        error = (fpnp.sum((numerc_sol - analyt_sol.value)**2)/len(numerc_sol))**(0.5)  
        #relative error
        rel_err= np.average(np.abs(numerc_sol-analyt_sol.value)/np.abs(numerc_sol), axis=0)

        errors[i] = error
        rel_errors[i] = rel_err

        #print(f'error {error}; rel_err {rel_err}')
        num_of_cells *= 2


    #%%
    from math import log as ln  # log is a fenics name too

    #%%  Show convergence accuracy
    log_nxvec = np.log(nxvec)
    log_errors_u = np.log(errors)
    fit_u = np.polyfit(log_nxvec, log_errors_u, 1)

    ax = axs[idx] 
    ax.plot(log_nxvec, log_errors_u, label="simulation errors")
    ax.plot(log_nxvec, np.poly1d(fit_u)(log_nxvec), linestyle="--",
            label="Line: {m:1.2f}x+{b:1.2f}".format(m=fit_u[0], b=fit_u[1]))
    ax.legend(loc="best")
    ax.set_xlabel("log(nx)")
    ax.set_ylabel("log(error)")
    ax.set_title(f'{delta_tp}')



    #%%
    rates_array = np.zeros(num_levels)
    for i in range(1, num_levels):
        Ei = errors[i]
        Eim1 = errors[i - 1]
        r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
        rates_array[i] = round(r, 2)

    rates[delta_tp] = rates_array


    rates_rel_array = np.zeros(num_levels)
    for i in range(1, num_levels):
        Ei = rel_errors[i]
        Eim1 = rel_errors[i - 1]
        r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
        rates_rel_array[i] = round(r, 2)

    rates_rel[delta_tp] = rates_rel_array

#%%  Salve img

fig.tight_layout()


from dedicate_code.config import reportings_dir

folder_name = 'delta'
report_folder = reportings_dir/folder_name
try:
    report_folder.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f"Folder {report_folder} is already there")
else:
    print(f"Folder {report_folder} was created")

file_name = f'new3_delta_accuc_Eps1_{tensor_product}'
fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")

#%%

#https://fenicsproject.org/pub/tutorial/html/._ftut1020.html

def compute_convergence_rates(u_e, f, u_D, kappa,
                              max_degree=3, num_levels=5):
    "Compute convergences rates for various error norms"

    h = {}  # discretization parameter: h[degree][level]
    E = {}  # error measure(s): E[degree][level][error_type]

    # Iterate over degrees and mesh refinement levels
    degrees = range(1, max_degree + 1)
    for degree in degrees:
        n = 8  # coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_levels):
            h[degree].append(1.0 / n)
            u = solver(kappa, f, u_D, n, n, degree, linear_solver='direct')
            errors = compute_errors(u_e, u)
            E[degree].append(errors)
            print('2 x (%d x %d) P%d mesh, %d unknowns, E1 = %g' %
              (n, n, degree, u.function_space().dim(), errors['u - u_e']))
            n *= 2

    # Compute convergence rates
    from math import log as ln  # log is a fenics name too
    etypes = list(E[1][0].keys())
    rates = {}
    for degree in degrees:
        rates[degree] = {}
        for error_type in sorted(etypes):
            rates[degree][error_type] = []
            for i in range(1, num_levels):
                Ei = E[degree][i][error_type]
                Eim1 = E[degree][i - 1][error_type]
                r = ln(Ei / Eim1) / ln(h[degree][i] / h[degree][i - 1])
                rates[degree][error_type].append(round(r, 2))

    return etypes, degrees, rates




# %%


#Function to calculate order of convergence  
def konvord(x,e):
    p = np.log(e[2:]/e[1:-1]) / np.log(e[1:-1]/e[:-2])
    mw = p.sum()/len(p)
    return mw #

#Function to calculate rate of convergence (for linear convergence)
def konvrate(x,e):
    n = len(e)
    k = np.arange(0,n) #array mit 0,1,2,3,...,n-1 (Iterationsschritte)
    fit = np.polyfit(k,np.log(e),1)
    L = np.exp(fit[0])
    return L


# %%
def rate(x, x_exact):
    e = [abs(x_ - x_exact) for x_ in x]
    q = [np.log(e[n+1]/e[n])/np.log(e[n]/e[n-1])
         for n in range(1, len(e)-1, 1)]
    return q

def print_rates(method, x, x_exact):
    q = ['%.2f' % q_ for q_ in rate(x, x_exact)]
    print(f'{method} :')
    for q_ in q:
        print(f'{q_}')