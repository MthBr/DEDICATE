"""
Version 1, 
Tester utils, for assessing convergenge accuracy
@author: enzo
"""

from dedicate_code.custom_funcs import benchmark
from dedicate_code.config import reportings_dir
from dedicate_code.setup_testSet import setup_dict


from dedicate_code.data_etl.etl_utils import gen_nonuniform_segment_delta, gen_nonuniform_chebyshev_delta, gen_x_nnunif_mesh, gen_x_mesh, gen_uniform_delta
from dedicate_code.feature_eng.solver_util import de_f_time


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

#%% Functions
def plot_final_funcs_x(num_of_cells, find_anlyt_error_func, timeDuration):
    # higher-order function
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(16, 9))#
    #1920x1080 (or 1080p)  figsize=(19.20,10.80)  (16, 9)   

    formula=setup_dict['eq_formula']
    specifc=setup_dict['text_long']
    title_txt = f'{formula} \n {specifc}'  #
    fig.suptitle(title_txt)


    delta_space = gen_uniform_delta(num_of_cells)
    mesh, x =gen_x_mesh(num_of_cells, delta_space)
    setup_dict['grid_spacing'] = 5*delta_space
    setup_dict['center_spacing'] = delta_space
    three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True

    error_u, analyt_unif = find_anlyt_error_func(mesh, three_time_phi[1][-1], timeDuration)
    ax=axs[2][0]
    ax.plot(x[0], analyt_unif, label="analytical")
    ax.plot(x[0], three_time_phi[1][-1], linestyle="--", label="u")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f'T={timeDuration}, err={error_u:.5f}')


    error_u, analyt_unif = find_anlyt_error_func(mesh, three_time_phi[1][-2], timeDuration/2)
    ax=axs[1][0]
    ax.plot(x[0], analyt_unif, label="analytical")
    ax.plot(x[0], three_time_phi[1][-2], linestyle="--", label="u")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f'T={timeDuration/2}, err={error_u:.5f}')

    error_u, analyt_unif = find_anlyt_error_func(mesh, three_time_phi[1][-3], timeDuration/10)
    ax=axs[0][0]
    ax.plot(x[0], analyt_unif, label="analytical")
    ax.plot(x[0], three_time_phi[1][-2], linestyle="--", label="u")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Uniform mesh {num_of_cells}\n"+f'T={timeDuration/10}, err={error_u:.5f}')



    
    delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells)
    mesh, x =gen_x_nnunif_mesh(delta_vect)
    setup_dict['grid_spacing'] = 5*delta_space
    setup_dict['center_spacing']  = delta_space
    three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
    
    errors_nu_seg, analyt_nonun = find_anlyt_error_func(mesh, three_time_phi[1][-1], timeDuration)
    ax=axs[2][1]
    ax.plot(x[0], analyt_nonun, label="analytical")
    ax.plot(x[0], three_time_phi[1][-1], linestyle="--", label="u")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f'T={timeDuration}, err={errors_nu_seg:.5f}')


    errors_nu_seg, analyt_nonun = find_anlyt_error_func(mesh, three_time_phi[1][-2], timeDuration/2)
    ax=axs[1][1]
    ax.plot(x[0], analyt_nonun, label="analytical")
    ax.plot(x[0], three_time_phi[1][-2], linestyle="--", label="u")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f'T={timeDuration/2}, err={errors_nu_seg:.5f}')

    errors_nu_seg, analyt_nonun = find_anlyt_error_func(mesh, three_time_phi[1][-3], timeDuration/10)
    ax=axs[0][1]
    ax.plot(x[0], analyt_nonun, label="analytical")
    ax.plot(x[0], three_time_phi[1][-2], linestyle="--", label="u")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Non-unif (segments) {num_of_cells} mesh\n"+f'T={timeDuration/10}, err={errors_nu_seg:.5f}')

    fig.tight_layout()


    #%% SAVE
    file_name = '2_' + setup_dict['compactxt'] + f'_{num_of_cells}'
    folder_name = setup_dict['id']
    report_folder = reportings_dir/folder_name
    try:
        report_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder {report_folder} is already there")
    else:
        print(f"Folder {report_folder} was created")


    # for pubblication: Nature: Postscript, Vector EPS or PDF format
    #maximum 300 p.p.i.; and min 300dpi
    fig.savefig(report_folder/(file_name+'.pdf'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.png'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.eps'), format='eps', dpi=300)
    fig.savefig(report_folder/(file_name+'.svg'), format='svg', dpi=300)

    return fig



def initializer(nxvec):
    import numpy as np
    errors_u = np.zeros(len(nxvec))
    errors_nu_seg = np.zeros(len(nxvec))
    return errors_u, errors_nu_seg

def init_pool(err_u, err_nu_seg):
    global errors_u, errors_nu_seg
    errors_u = err_u
    errors_nu_seg = err_nu_seg


def gen_errs(indx, num_of_cells, find_anlyt_error_func, timeDuration):
    delta_space = gen_uniform_delta(num_of_cells)
    mesh, x =gen_x_mesh(num_of_cells, delta_space)
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing'] = delta_space
    three_time_phi = de_f_time(indx, setup_dict, mesh, None, timeDuration)  #False  True
    errors_u, _ = find_anlyt_error_func(mesh, three_time_phi[1][-1], timeDuration)

    delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells)
    mesh, x =gen_x_nnunif_mesh(delta_vect)
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing']  = delta_space
    three_time_phi = de_f_time(indx, setup_dict, mesh, None, timeDuration)  #False  True
    errors_nu_seg, _ = find_anlyt_error_func(mesh, three_time_phi[1][-1], timeDuration)

    return indx, errors_u, errors_nu_seg

def merge_result_err(result):
    indx = result[0]
    global errors_u, errors_nu_seg
    errors_u[indx] = result[1]
    errors_nu_seg[indx] = result[2]

@benchmark
def error_multicore(nxvec, find_anlyt_error_func, timeDuration):
    import multiprocessing as mp
    from functools import partial
    global errors_u, errors_nu_seg
    errors_u, errors_nu_seg = initializer(nxvec)
    pool = mp.Pool(mp.cpu_count()-1, init_pool(errors_u, errors_nu_seg))
    for indx, num_of_cells in enumerate(nxvec):
        pool.apply_async(gen_errs, 
        args=(indx, num_of_cells, find_anlyt_error_func, timeDuration), 
        callback=merge_result_err)
    pool.close()
    pool.join()
    return errors_u, errors_nu_seg


@benchmark
def error_onecore(nxvec, find_anlyt_error_func, timeDuration):
    global errors_u, errors_nu_seg
    errors_u, errors_nu_seg = initializer(nxvec)
    for indx, num_of_cells in enumerate(nxvec):
        merge_result_err(gen_errs(indx, num_of_cells, find_anlyt_error_func, timeDuration))
    return errors_u, errors_nu_seg



def calulate_all_errors(nxvec, find_anlyt_error_func, timeDuration):
    return error_multicore(nxvec, find_anlyt_error_func, timeDuration)


def  calulate_all_errors_bkup(nxvec, find_anlyt_error_func, timeDuration):
    # higher-order function
    import numpy as np

    errors_u = np.zeros(len(nxvec))
    errors_nu_seg = np.zeros(len(nxvec))
    for indx, num_of_cells in enumerate(nxvec):
        delta_space = gen_uniform_delta(num_of_cells)
        mesh, x =gen_x_mesh(num_of_cells, delta_space)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing'] = delta_space
        three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
        errors_u[indx], _ = find_anlyt_error_func(mesh, three_time_phi[1][-1], timeDuration)
        
        delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells)
        mesh, x =gen_x_nnunif_mesh(delta_vect)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing']  = delta_space
        three_time_phi = de_f_time(1, setup_dict, mesh, None, timeDuration)  #False  True
        errors_nu_seg[indx], _ = find_anlyt_error_func(mesh, three_time_phi[1][-1], timeDuration)
        
    return  errors_u, errors_nu_seg




def plot_conv_acc(timeDuration, nxvec, errors_u, errors_nu_seg):
    import numpy as np
    import matplotlib.pyplot as plt

    #%%  Show convergence accuracy
    log_nxvec = np.log(nxvec)
    log_errors_u = np.log(errors_u)
    log_errors_nu_seg = np.log(errors_nu_seg)
    fit_u = np.polyfit(log_nxvec, log_errors_u, 1)
    fit_nu_seg = np.polyfit(log_nxvec, log_errors_nu_seg, 1)

    fig, axs = plt.subplots(1, 2, figsize=(16, 9))#
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


    fig.tight_layout()

    #%% SAVE
    file_name = '2_'+setup_dict['compactxt']
    folder_name = setup_dict['id']
    report_folder = reportings_dir/folder_name
    try:
        report_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder {report_folder} is already there")
    else:
        print(f"Folder {report_folder} was created")

    # for pubblication: Nature: Postscript, Vector EPS or PDF format
    #maximum 300 p.p.i.; and min 300dpi
    fig.savefig(report_folder/(file_name+'.pdf'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.png'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.eps'), format='eps', dpi=300)
    fig.savefig(report_folder/(file_name+'.svg'), format='svg', dpi=300)

    return fig
