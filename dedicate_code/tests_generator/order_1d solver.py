#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))


import os
os.environ["FIPY_SOLVERS"] = "scipy" #pyamgx petsc scipy no-pysparse trilinos  pysparse pyamg
os.environ["FIPY_VERBOSE_SOLVER"] = "1" # 1:True # Only for TESTING
#print('Verbose ' + os.environ.get('FIPY_VERBOSE_SOLVER'))
os.environ["OMP_NUM_THREADS"]= "1"

import matplotlib.pyplot as plt
import matplotlib as mpl
axtickfsize = 12
labelfsize = 9
legfsize = labelfsize - 2
txtfsize = labelfsize - 2
lwidth = 3
markersize = 5
markeredgewidth = 0.1
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = labelfsize
mpl.rcParams['xtick.labelsize'] = axtickfsize
mpl.rcParams['ytick.labelsize'] = axtickfsize
mpl.rcParams['font.size'] = txtfsize
mpl.rcParams["figure.titlesize"] = 26
mpl.rcParams["figure.titleweight"] = 'regular'
mpl.rcParams['legend.fontsize'] = legfsize
mpl.rcParams['lines.linewidth'] = lwidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = markeredgewidth


setup_dict = {}
setup_dict['dimensions'] = 1
setup_dict['D'] = -1
setup_dict['convection'] = False
setup_dict['initial'] = 'None'
setup_dict['source'] = 'delta0.5'
setup_dict['boundaryConditions'] = 'Dirichlet' 


import numpy as np
import matplotlib.pyplot as plt
import fipy as fp


from dedicate_code.data_etl.etl_utils import array_from_mesh , gen_x_mesh
from dedicate_code.data_etl.etl_utils import gen_uniform_delta
from dedicate_code.feature_eng.solver4stationary_util import single_solve

#%%
test = 'Tiol' #'l-1-1-1d',  cos-1-1d
delta_type = 'cos-2-2d'

# = ['l-1-0-1d', 'l-1-1-2d', 'l-1-2-2d', 'l-2-2-1d', 'l-2-2-2d', 'l-2-3-1d', 'l-2-3-2d', 'l-2-5-1d', 'l-2-5-2d']

#test = 12 #'l-1-1-1d',
#delta_types = ['l-1-0-1d-s2', 'l-1-1-2d-s2', 'l-1-2-2d-s2', 'l-2-2-1d-s2', 'l-2-2-2d-s2', 'l-2-3-1d-s2', 'l-2-3-2d-s2', 'l-2-5-1d-s2', 'l-2-5-2d-s2']


#test = 2
#delta_types =['l-1-1-1d', 'cos-1-1d', 'cos-2-1d', 'cos-2-2d']

#test = 22
#delta_types =['l-1-1-1d-s2', 'cos-1-1d-s2', 'cos-2-1d-s2', 'cos-2-2d-s2', 'p-cos-s2', 'p*cos-s25']


#test = 3
#delta_types =['p2h-s1', 'p4h-s2', 'p3-s15', 'p4-s2', 'p*2-s15','p*3-s2','p*4-s25', 'p-cubic-s2', 'p-LL-s2']

#test = 4
#delta_types =['p6-s3', 'pg5-s25', 'pg6-s3'] #g: gaussian like

solver_types =[
#'agg_cheb4', #Diverges23
'AGGREGATION_DILU',
'AGGREGATION_GS',
'AGGREGATION_JACOBI',#Diverges23
'AGGREGATION_LOW_DEG_BJ',#Diverges2
'AGGREGATION_LOW_DEG_DILU',
'AGGREGATION_LOW_DEG_GS',
'AGGREGATION_MULTI_PAIRWISE',#Diverges2
'AGGREGATION_THRUST_BJ',
'AGGREGATION_THRUST_DILU',
'AGGREGATION_THRUST_GS',#Diverges3
'AMG_AGGRREGATION_CG',
'AMG_CLASSICAL_AGGRESSIVE_CHEB_L1_TRUNC',
'AMG_CLASSICAL_AGGRESSIVE_L1',
'AMG_CLASSICAL_AGGRESSIVE_L1_TRUNC',
'AMG_CLASSICAL_CG',
'AMG_CLASSICAL_CGF',
'AMG_CLASSICAL_L1_AGGRESSIVE_HMIS',
'AMG_CLASSICAL_L1_TRUNC',
'AMG_CLASSICAL_PMIS',
'CG_DILU',#Diverges
#'CHEB_SOLVER_NOPREC', # Incorrect amgx configuration provided.
'CLASSICAL_CG_CYCLE',
'CLASSICAL_CGF_CYCLE',
'CLASSICAL_F_CYCLE',
'CLASSICAL_V_CYCLE',
'CLASSICAL_W_CYCLE',
'F',
'FGMRES',
'FGMRES_AGGREGATION',
'FGMRES_AGGREGATION_DILU',#Diverges
'FGMRES_AGGREGATION_JACOBI',
'FGMRES_CLASSICAL_AGGRESSIVE_HMIS',
'FGMRES_CLASSICAL_AGGRESSIVE_PMIS',
'FGMRES_NOPREC',#Diverges23
'GMRES',#Diverges23
'GMRES_Agg',
'GMRES_Agg2',
'GMRES_AMG_D2',
'IDR_DILU',
'IDRMSYNC_DILU',
'JACOBI',#Diverges23
'PBICGSTAB',
'PBICGSTAB_AGGREGATION_W_JACOBI',
'PBICGSTAB_AGG_W_JAC',
'PBICGSTAB_CLASSICAL_JACOBI',
'PBICGSTAB_NOPREC',#Diverges
'PBICGSTAB_W',
'PCG_AGGREGATION_JACOBI',
'PCG_CLASSICAL_F_JACOBI',
'PCG_CLASSICAL_V_JACOBI',
'PCG_CLASSICAL_W_JACOBI',
'PCG_DILU',
'PCG_F',
'PCGF_CLASSICAL_F_JACOBI',
'PCGF_CLASSICAL_V_JACOBI',
'PCGF_CLASSICAL_W_JACOBI',
'PCGF_F',
'PCGF_V',
'PCGF_W',
'PCG_NOPREC',#Diverges
'PCG_V',
'PCG_W',
'V',
'V-cheby-aggres-L1-trunc',
'V-cheby-aggres-L1-trunc-userLambda',
'V-cheby_poly-smoother',
'V-cheby-smoother',
'W'
]


# solver_types =[
# "AMG_CLASSICAL_AGGRESSIVE_CHEB_L1_TRUNC",
# 'AMG_CLASSICAL_AGGRESSIVE_L1',
# 'AMG_CLASSICAL_AGGRESSIVE_L1_TRUNC',#bst?
# 'FGMRES_CLASSICAL_AGGRESSIVE_HMIS',
# 'GMRES_Agg',
# 'PCG_enx_V_AGGRESSIVE_CHEB_L1',
# 'AMG_CLASSICAL_PMIS',
# 'V-cheby-aggres-L1-trunc', #bst?
# 'V-cheby-aggres-L1-trunc-userLambda',  #bst?
# ]


from dedicate_code.tests_generator.utils_delta_stats import plot_nsave_solution

#%%

fpnp = fp.numerix
tensor_product = 'tensor'

ksup = 1.0

nxvec = 2 ** np.arange(5,12)  #7-12 coarsest mesh division
num_levels= len(nxvec)

n_plots = len(solver_types) #number_of_subplots
Cols = int(n_plots**0.5)
# Compute Rows required
Rows = n_plots // Cols 
Rows += n_plots % Cols

fig, axs_m = plt.subplots(Rows, Cols, figsize=(16, 9)) #, sharex=True, sharey=True
axs = axs_m.flatten()

rates = {}
rates_rel = {}

for idx, solver_tp  in enumerate(solver_types):
    
    errors = np.zeros(num_levels)# error measure(s): E[level][error_type]
    rel_errors = np.zeros(num_levels)
    h_deltas = np.zeros(num_levels) # discretization parameter: h[level]

    print(solver_tp)

    for i in range(num_levels):
        num_of_cells = nxvec[i]
        delta_space = gen_uniform_delta(num_of_cells, length=1)
        mesh =gen_x_mesh(num_of_cells, delta_space, start=0)
        xy = array_from_mesh(setup_dict['dimensions'], mesh)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing'] = delta_space
        setup_dict['delta_type'] = delta_type
        epsilon_delta = ksup * delta_space #* 1.5 # 0.1 #  delta_space
        setup_dict['epsilon_delta'] = epsilon_delta  # 
        setup_dict['delta_product'] = tensor_product
        #setup_dict['solver_type'] = 'petci-PCG'
        #setup_dict['precon'] = 'jacobi'
        setup_dict['solver_type'] = 'AmgX' #AmgX petci-  scip-PCG  PCG CGS GMRE LU   
        setup_dict['precon'] = solver_tp #mh  'sor' jacobi ilu
        #ok1:  sc-CGS + jacobi    sc-PCG + jacobi 
        #ok2:  NOOOOOOOOOOOOO     sc-PCG + jacobi 
        #PCG_AGGREGATION_JACOBI

        print(i)

        h_deltas[i]=delta_space

        numerc_sol = single_solve(setup_dict, mesh, None)  #False  True
        analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
        #X = np.linalg.norm([mesh.x.value], ord=2, axis=0)
        X = mesh.x.value
        mask1 = (X <= 0.5)
        s1 = np.zeros_like(X)
        s1_m = X[mask1]
        s1[mask1] = s1_m*(1-0.5)
        s1[~mask1] = 0.0
        mask2 = ((0.5 < X) &  (X<= 1.0))
        s2 = np.zeros_like(X)
        s2_m = X[mask2]
        s2[mask2] = 0.5*(1-s2_m)
        s2[~mask2] = 0.0
        analyt_sol[:] =  s1+s2

        error = (fpnp.sum((numerc_sol - analyt_sol.value)**2)/len(numerc_sol))**(0.5)  
        #relative error
        rel_err= np.average(np.abs(numerc_sol-analyt_sol.value)/np.abs(numerc_sol), axis=0)

        print(i)

        title_str = f"{solver_tp}  semi-support:{epsilon_delta} \n err={error:.5e}, mesh len 1d: {num_of_cells}"
        file_name = f"delta {setup_dict['dimensions']}d_{solver_tp}_mes_{num_of_cells}_suport{int(epsilon_delta*1000)}"
        placeholder_fig = plot_nsave_solution(xy[0], analyt_sol.value, numerc_sol, title_str, file_name)
        placeholder_fig.clear()


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
    ax.set_title(f'{solver_tp}')



    #%%
    rates_array = np.zeros(num_levels)
    for i in range(1, num_levels):
        Ei = errors[i]
        Eim1 = errors[i - 1]
        r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
        rates_array[i] = round(r, 2)
    rates_array[0] = np.mean(rates_array[1:])
    rates[solver_tp] = rates_array.tolist()


    rates_rel_array = np.zeros(num_levels)
    for i in range(1, num_levels):
        Ei = rel_errors[i]
        Eim1 = rel_errors[i - 1]
        r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
        rates_rel_array[i] = round(r, 2)

    rates_rel_array[0] = np.mean(rates_rel_array[1:])
    

    rates_rel[solver_tp] = rates_rel_array.tolist()

#%%  Salve img

fig.tight_layout()


from dedicate_code.config import reportings_dir

folder_name = 'delta'
sub_folder_name = f"delta{test}_D{setup_dict['dimensions']}_suport{ksup}_{int(epsilon_delta*1000)}_delta_tp{delta_type}"
report_folder = reportings_dir/folder_name/sub_folder_name
try:
    report_folder.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f"Folder {report_folder} is already there")
else:
    print(f"Folder {report_folder} was created")

file_name = f'1D_delta_accuc_Eps1_{tensor_product}_ test_{test}'
fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")

# %%

file_name1 = 'rates.json'
import json
# Serialize data into file:
json.dump( rates, open( report_folder/file_name1, 'w' ) )
file_name2 = 'rates_rel.json'
json.dump( rates_rel, open( report_folder/file_name2, 'w' ) )


# %%

# %%
