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
labelfsize = 7
legfsize = labelfsize - 5
txtfsize = labelfsize - 5
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
setup_dict['dimensions'] = 2
setup_dict['D'] = -1.0
setup_dict['convection'] = False
setup_dict['initial'] = 'None'
setup_dict['source'] = 'delta'
setup_dict['boundaryConditions'] = 'Dirichlet' 






import numpy as np
import matplotlib.pyplot as plt
import fipy as fp


from dedicate_code.data_etl.etl_utils import array_from_mesh , gen_xy_mesh, gen_xy_nnunif_mesh
from dedicate_code.data_etl.etl_utils import gen_uniform_delta, gen_nonuniform_central_segment_delta
from dedicate_code.feature_eng.solver4stationary_util import single_solve

#%%
seq='Power2'
nxvec = 2 ** np.arange(5,12)  #7-12 coarsest mesh division


##seq='Seq22'
#nxvec = 2 * np.arange(512,521)  #7-12 coarsest mesh division

#seq='Seq2'
#nxvec = 2 * np.arange(256,267)  #7-12 coarsest mesh division
#nxvec = 2 * np.arange(1050,1055)  #7-12 coarsest mesh division

#seq='Seq16'
#nxvec = 2* np.arange(1025,1150, 16)

#seq='Seq3'
#nxvec = (2 * np.arange(256,267))+1  #7-12 coarsest mesh division

#seq='SeqE'
#nxvec = np.arange(2024,2035)  #7-12 coarsest mesh division




#test = '0'

# delta_types = ['l-1-0-1d', 'l-1-1-2d', 'l-1-2-2d', 'l-2-2-1d', 
#     'l-2-2-2d', 'l-2-3-1d', 'l-2-3-2d', 'l-2-5-1d', 'l-2-5-2d',
#     'l-1-1-1d', 'cos-1-1d', 'cos-2-1d', 'cos-2-2d', 'l-1-1-1d-s2', 'cos-1-1d-s2',
#      'cos-2-1d-s2', 'p-cos-s2', 'p*cos-s25',
#      'p2h-s1', 'p4h-s2', 'p3-s15', 'p4-s2', 'p*2-s15','p*3-s2','p*4-s25', 'p-cubic-s2', 'p-LL-s2',
#      'pg5-s25', 'pg6-s3'
#     ]


test = '0tesadf'
delta_types = [ 'adf-0-z0', 'adf-1-z0', 'adf-2-z0', 'adf-3-z0', 'adf-4-z0']


#test = '1' #'l-1-1-1d',
#delta_types = ['l-1-0-1d', 'l-1-1-2d']
#delta_types = ['l-1-0-1d', 'l-1-1-2d', 'l-1-2-2d', 'l-2-2-1d', 'l-2-2-2d', 'l-2-3-1d', 'l-2-3-2d', 'l-2-5-1d', 'l-2-5-2d']




#test = '2' #'2FullTest' #'l-1-1-1d',
#['l-1-1-1d', 'cos-1-1d', 'cos-2-1d', 'cos-2-2d', 'l-1-1-1d-s2', 'cos-1-1d-s2', 'cos-2-1d-s2', 'p-cos-s2', 'p*cos-s25']

#test = 12 #'l-1-1-1d',
#delta_types = ['l-1-0-1d-s2', 'l-1-1-2d-s2', 'l-1-2-2d-s2', 'l-2-2-1d-s2', 'l-2-2-2d-s2', 'l-2-3-1d-s2', 'l-2-3-2d-s2', 'l-2-5-1d-s2', 'l-2-5-2d-s2']
#test = 2
#delta_types =['l-1-1-1d', 'cos-1-1d', 'cos-2-1d', 'cos-2-2d']
#test = 22
#delta_types =['l-1-1-1d-s2', 'cos-1-1d-s2', 'cos-2-1d-s2', 'cos-2-2d-s2', 'p-cos-s2', 'p*cos-s25']


#test = '3'
#delta_types =['p2h-s1', 'p4h-s2', 'p3-s15', 'p4-s2', 'p*2-s15','p*3-s2','p*4-s25', 'p-cubic-s2', 'p-LL-s2']



#test = '32'
#delta_types =['p4h-s2','p*3-s2','p*4-s25',]



#test = '44NonUnif'
#delta_types =[ 'pg6-s3'] #g: gaussian like  'p4-s2', 'pg5-s25',
#delta_types =['p6-s3', 'pg5-s25', 'pg6-s3'] #g: gaussian like

num_levels= len(nxvec)


#setup_dict['solver_type'] = 'petci-PCG'
#setup_dict['precon'] = 'jacobi'
setup_dict['solver_type'] = 'AmgX' #AmgX petci-  scip-PCG  PCG CGS GMRE LU   
setup_dict['precon'] = 'V-cheby-aggres-L1-trunc' #mh  'sor' jacobi ilu
#ok1:  sc-CGS + jacobi    sc-PCG + jacobi 
#ok2:  NOOOOOOOOOOOOO     sc-PCG + jacobi 
#PCG_AGGREGATION_JACOBI
#AMG_CLASSICAL_AGGRESSIVE_L1_TRUNC
#V-cheby-aggres-L1-trunc


fpnp = fp.numerix
den = 2*fpnp.pi*setup_dict['D']
delta_product = 'tensor' # 'radial' 'scaled_tensor' tensor


ksup_arr = [1.0, 1.5, 2.0]
num_k = len(ksup_arr)


rates = []
rates_rel = []

rates.append(['delta', 'k', f"mean {seq} {setup_dict['precon']}", *nxvec])
rates_rel.append(['delta', 'k', f"mean {seq} {setup_dict['precon']}", *nxvec])



#%%
for idx, delta_tp  in enumerate(delta_types):
    errors = np.zeros((num_levels, num_k))# error measure(s): E[level][error_type]
    rel_errors = np.zeros((num_levels, num_k))
    h_deltas = np.zeros(num_levels) # discretization parameter: h[level]
    print(f"************** \n {delta_tp} \n *****************************")

    for i in range(num_levels):
        num_of_cells = nxvec[i]
        delta_space = gen_uniform_delta(num_of_cells, length=3.0)
        #delta_vect, delta_space = gen_nonuniform_central_segment_delta(num_of_cells, length=2.0)
        mesh =gen_xy_mesh(num_of_cells, delta_space, start=-1.5)
        #mesh =gen_xy_nnunif_mesh(delta_vect, start=-1.0)
        xy = array_from_mesh(setup_dict['dimensions'], mesh, num_of_cells)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing'] = delta_space
        setup_dict['delta_type'] = delta_tp
        h_deltas[i]=delta_space


        for j, ksup in enumerate(ksup_arr):
            epsilon_delta = ksup * delta_space #* 1.5 # 0.1 #  delta_space
            setup_dict['epsilon_delta'] = epsilon_delta  # 
            setup_dict['delta_product'] = delta_product

            #print(i)
            numerc_sol = single_solve(setup_dict, mesh, None)  #False  True


            analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
            X = np.linalg.norm([mesh.x.value, mesh.y.value], ord=2, axis=0)
            l0=1.5  #np.sqrt(1.5**2+1.5**2) #1.5
            analyt_sol[:] =  (fpnp.log(X/l0) / den)
            analyt_sol= analyt_sol.value

            where_inf = np.isinf(analyt_sol)
            #len(analyt_sol[~ where_inf])
            #len(numerc_sol[~ where_inf])
            numerc_sol = numerc_sol[~ where_inf]
            analyt_sol = analyt_sol[~ where_inf]


            error = (fpnp.sum((numerc_sol - analyt_sol)**2)/len(numerc_sol))**(0.5)  
            #relative error
            rel_err= np.average(np.abs(numerc_sol-analyt_sol)/np.abs(numerc_sol), axis=0)

            #print(i)


            #numerc_2d=numerc_sol.reshape((num_of_cells, num_of_cells))
            #analyt_2d = analyt_sol.reshape((num_of_cells, num_of_cells))
            #title_str = f'{delta_tp}  semi-support:{epsilon_delta} \n err={error:.5e}, mesh len 1d: {num_of_cells}_{ksup:.1f}'
            #file_name = f'delta 1d_{delta_tp}_mes_{num_of_cells}_suport{int(epsilon_delta*1000)}_{int(ksup*10)}'
            #placeholder_fig = plot_nsave_solution(xy[0], analyt_2d[num_of_cells//2], numerc_2d[num_of_cells//2], title_str, file_name)
            #placeholder_fig.clear()

            errors[i][j] = error
            rel_errors[i][j] = rel_err

            #print(f'error {error}; rel_err {rel_err}')
            #num_of_cells *= 2


        #%%
    from math import log as ln  # log is a fenics name too

    for j, ksup in enumerate(ksup_arr):

        #%%
        rates_array = np.zeros(num_levels)
        for i in range(1, num_levels):
            Ei = errors[i][j]
            Eim1 = errors[i - 1][j]
            r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
            rates_array[i] = round(r, 3)
        
        rates_array[0] = np.format_float_positional(np.mean(rates_array[1:]),2)
        rates_array = [delta_tp, ksup, *rates_array[1:].tolist()]
        rates.append(rates_array)

        rates_rel_array = np.zeros(num_levels)
        for i in range(1, num_levels):
            Ei = rel_errors[i][j]
            Eim1 = rel_errors[i - 1][j]
            r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
            rates_rel_array[i] = round(r, 3)
        rates_rel_array[0] = np.format_float_positional(np.mean(rates_rel_array[1:]),2)

        rates_rel_array = [delta_tp, ksup, *rates_rel_array.tolist()]
        rates_rel.append(rates_rel_array)



print('Ended!!1')
#%% 

from dedicate_code.config import reportings_dir

folder_name = f"delta{setup_dict['dimensions']}"
#sub_folder_name = f"delta{test}_D{setup_dict['dimensions']}_suport{ksup}_{int(epsilon_delta*1000)}_sol_{setup_dict['solver_type']}_{setup_dict['precon']}"
report_folder = reportings_dir/folder_name#/sub_folder_name
try:
    report_folder.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f"Folder {report_folder} is already there")
else:
    print(f"Folder {report_folder} was created")





file_name1 = f"rates_tst{test}_{seq}_{delta_product}_sol_{setup_dict['solver_type']}_{setup_dict['precon']}.csv"

import csv

with open(report_folder/file_name1, "w", newline="") as f:
    writer = csv.writer(f, delimiter=',',  quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rates)



file_name2 = f"rates_rel_tst{test}_{seq}_{delta_product}_sol_{setup_dict['solver_type']}_{setup_dict['precon']}.csv"
with open(report_folder/file_name2, "w", newline="") as f:
    writer = csv.writer(f, delimiter=',',  quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rates_rel)



# %%
print(file_name2)


# %%
