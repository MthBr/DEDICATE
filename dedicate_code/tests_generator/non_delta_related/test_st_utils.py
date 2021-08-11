"""
Version 1, 
Tester utils, for assessing convergenge accuracy
@author: enzo
"""

from dedicate_code.custom_funcs import benchmark
from dedicate_code.config import reportings_dir


from dedicate_code.data_etl.etl_utils import array_from_mesh, gen_xyz_mesh, gen_xyz_nnunif_mesh, gen_xy_nnunif_mesh, gen_xy_mesh
from dedicate_code.data_etl.etl_utils import gen_nonuniform_segment_delta, gen_uniform_delta
from dedicate_code.feature_eng.solver4stationary_util import solve, single_solve


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
mpl.rcParams["figure.titlesize"] = 26
mpl.rcParams["figure.titleweight"] = 'semibold'
mpl.rcParams['legend.fontsize'] = legfsize
mpl.rcParams['lines.linewidth'] = lwidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = markeredgewidth
LEN=2
STR=-1

#%% Functions
def plot_final_funcs_xy(num_of_cells, find_anlyt_error_func, setup_dict):
    # higher-order function
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=(16, 9))#
    #1920x1080 (or 1080p)  figsize=(19.20,10.80)  (16, 9)   
    nxy = num_of_cells

    formula=setup_dict['eq_formula']
    specifc=setup_dict['eq_formula2']
    title_txt = f'{formula} \n {specifc}'  #
    fig.suptitle(title_txt)


    delta_space = gen_uniform_delta(num_of_cells, length=2)
    mesh =gen_xy_mesh(num_of_cells, delta_space, start=-1)
    xy = array_from_mesh(2, mesh)
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing'] = delta_space
    final_phi = single_solve(setup_dict, mesh, None)  #False  True

    error_u, analyt_unif = find_anlyt_error_func(mesh, final_phi)
    ax=axs[0][0]
    final_phi_2d=final_phi.reshape((nxy, nxy))
    analyt_unif_2d = analyt_unif.reshape((nxy, nxy))

    ax.plot(xy[0], analyt_unif_2d[nxy//2], label="analytical")
    ax.plot(xy[0], final_phi_2d[nxy//2], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("x(0.5)")
    ax.set_ylabel("h")
    ax.set_title(f'err={error_u:.5f}, mesh len 1d: {num_of_cells}; 2d:{num_of_cells**2}')


    ax=axs[1][0]

    ax.plot(xy[1], analyt_unif_2d[:,nxy//2], label="analytical")
    ax.plot(xy[1], final_phi_2d[:,nxy//2], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("y(0.5)")
    ax.set_ylabel("h")
    

    x_labels = [int(j) for j in range(STR,STR+LEN+1)]
    extend_img=[STR,STR+LEN,STR,STR+LEN]
    ax=axs[0][1]
    ax.imshow(analyt_unif_2d, cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'analytical')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)

    ax=axs[1][1]
    ax.imshow(final_phi_2d, origin="lower", cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'numeric')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)


    #########################################################


    delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells, length=2)
    mesh =gen_xy_nnunif_mesh(delta_vect, start=-1.0)
    xy = array_from_mesh(2, mesh, len(delta_vect))
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing']  = delta_space
    final_phi = single_solve(setup_dict, mesh, None)  #False  True
    
    errors_nu_seg, analyt_nonun = find_anlyt_error_func(mesh, final_phi)
    final_phi_2d=final_phi.reshape((nxy, nxy))
    analyt_nnunif_2d = analyt_nonun.reshape((nxy, nxy))


    ax=axs[2][0]
    ax.plot(xy[0], analyt_nnunif_2d[nxy//2], label="analytical")
    ax.plot(xy[0], final_phi_2d[nxy//2], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("h")
    ax.set_title(f'err={errors_nu_seg:.5f}')

    ax=axs[3][0]
    ax.plot(xy[1], analyt_nnunif_2d[:,nxy//2], label="analytical")
    ax.plot(xy[1], final_phi_2d[:,nxy//2], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("y")
    ax.set_ylabel("h")


    
    from matplotlib.image import NonUniformImage
    interp = 'nearest'  # nearest bilinear

    ax=axs[2][1]
    im = NonUniformImage(ax, interpolation=interp, extent=extend_img, cmap=plt.get_cmap('viridis'))
    im.set_data(xy[0], xy[1], final_phi_2d)
    ax.images.append(im)
    ax.set_aspect('equal')
    ax.set_title(f'analytical')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)

    ax=axs[3][1]
    im = NonUniformImage(ax, interpolation=interp, origin="lower", extent=extend_img, cmap=plt.get_cmap('viridis'))
    im.set_data(xy[0], xy[1], analyt_nnunif_2d)
    ax.images.append(im)
    ax.set_aspect('equal')
    ax.set_title(f'numeric')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)


    #%% SAVE
    fig.tight_layout()

    file_name = '2_' + setup_dict['compactxt_eq'] + f'_{num_of_cells}'
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







def plot_final_funcs_xyz_unif(num_of_cells, find_anlyt_error_func, setup_dict):
    # higher-order function
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(16, 9))#
    #1920x1080 (or 1080p)  figsize=(19.20,10.80)  (16, 9)   
    nxy = num_of_cells

    formula=setup_dict['eq_formula']
    specifc=setup_dict['eq_formula2']
    title_txt = f'{formula} \n {specifc}'  #
    fig.suptitle(title_txt)


    delta_space = gen_uniform_delta(num_of_cells, length=2)
    mesh =gen_xyz_mesh(num_of_cells, delta_space, start=-1)
    xyz = array_from_mesh(3, mesh)
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing'] = delta_space
    final_phi = single_solve(setup_dict, mesh, None)  #False  True

    error_u, analyt_unif = find_anlyt_error_func(mesh, final_phi)
    
    final_phi_3d=final_phi.reshape((nxy, nxy, nxy))
    analyt_unif_3d = analyt_unif.reshape((nxy, nxy, nxy))

    ax=axs[0][0]
    ax.plot(xyz[0], analyt_unif_3d[nxy//2, nxy//2, :], label="analytical")
    ax.plot(xyz[0], final_phi_3d[nxy//2, nxy//2, :], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("z ; xy(0.5)")
    ax.set_ylabel("h")
    ax.set_title(f'err={error_u:.5f}, mesh len 1d: {num_of_cells}; 2d:{num_of_cells**2}')


    ax=axs[0][1]
    ax.plot(xyz[1], analyt_unif_3d[:,nxy//2, nxy//2], label="analytical")
    ax.plot(xyz[1], final_phi_3d[:,nxy//2, nxy//2], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("x; yz(0.5)")
    ax.set_ylabel("h")
    
    ax=axs[0][2]
    ax.plot(xyz[1], analyt_unif_3d[nxy//2,:,nxy//2], label="analytical")
    ax.plot(xyz[1], final_phi_3d[nxy//2,:,nxy//2], linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("y; xz(0.5)")
    ax.set_ylabel("h")


    x_labels = [int(j) for j in range(STR,STR+LEN+1)]
    extend_img=[STR,STR+LEN,STR,STR+LEN]

    ax=axs[1][0]
    ax.imshow(analyt_unif_3d[nxy//2, :, :], cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'analytical x(0.5)')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    ax.set_xlabel("y")
    ax.set_ylabel("z")

    ax=axs[2][0]
    ax.imshow(final_phi_3d[nxy//2, :, :], origin="lower", cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'numeric x(0.5)')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    ax.set_xlabel("y")
    ax.set_ylabel("z")

    ax=axs[1][1]
    ax.imshow(analyt_unif_3d[:, nxy//2, :], cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'analytical y(0.5)')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax=axs[2][1]
    ax.imshow(final_phi_3d[:, nxy//2, :], origin="lower", cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'numeric y(0.5)')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    ax.set_xlabel("x")
    ax.set_ylabel("z")



    ax=axs[1][2]
    ax.imshow(analyt_unif_3d[:, : , nxy//2], cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'analytical z(0.5)')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax=axs[2][2]
    ax.imshow(final_phi_3d[:, :, nxy//2], origin="lower", cmap=plt.get_cmap('viridis'), extent=extend_img)
    ax.set_title(f'numeric z(0.5)')
    ax.set_xticks(x_labels)
    ax.set_yticks(x_labels)
    ax.set_xlabel("x")
    ax.set_ylabel("y")




    #%% SAVE
    fig.tight_layout()

    file_name = '2_' + setup_dict['compactxt_eq'] + f'_{num_of_cells}'
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
    #fig.savefig(report_folder/(file_name+'.pdf'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")
    #fig.savefig(report_folder/(file_name+'.png'), bbox_inches="tight")
    #fig.savefig(report_folder/(file_name+'.eps'), format='eps', dpi=300)
    #fig.savefig(report_folder/(file_name+'.svg'), format='svg', dpi=300)

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


def gen_errs(indx, num_of_cells, find_anlyt_error_func, setup_dict):
    dims = setup_dict['dimensions']

    delta_space = gen_uniform_delta(num_of_cells, LEN)
    if dims == 3:
        mesh =gen_xyz_mesh(num_of_cells, delta_space, STR)
    elif dims == 2:
        mesh =gen_xy_mesh(num_of_cells, delta_space, STR)
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing'] = delta_space
    final_phi  = single_solve(setup_dict, mesh, None)  #False  True
    errors_u, _ = find_anlyt_error_func(mesh, final_phi)

    delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells, LEN)
    if dims == 3:
        mesh =gen_xyz_nnunif_mesh(delta_vect, STR)
    elif dims == 2:
        mesh =gen_xy_nnunif_mesh(delta_vect, STR)
    setup_dict['grid_spacing'] = delta_space
    setup_dict['center_spacing']  = delta_space
    final_phi = single_solve(setup_dict, mesh, None)  #False  True
    errors_nu_seg, _ = find_anlyt_error_func(mesh, final_phi)

    return indx, errors_u, errors_nu_seg

def merge_result_err(result):
    indx = result[0]
    global errors_u, errors_nu_seg
    errors_u[indx] = result[1]
    errors_nu_seg[indx] = result[2]

@benchmark
def error_multicore(nxvec, find_anlyt_error_func, setup_dict):
    import multiprocessing as mp
    from functools import partial
    global errors_u, errors_nu_seg
    errors_u, errors_nu_seg = initializer(nxvec)
    dims = setup_dict['dimensions']
    if dims == 3:
        cpus = mp.cpu_count()-3
    elif dims == 2:
        cpus =mp.cpu_count()-2
    else:
        cpus = mp.cpu_count()-1
    pool = mp.Pool(cpus, init_pool(errors_u, errors_nu_seg))
    for indx, num_of_cells in enumerate(nxvec):
        pool.apply_async(gen_errs, 
        args=(indx, num_of_cells, find_anlyt_error_func, setup_dict), 
        callback=merge_result_err)
    pool.close()
    pool.join()
    return errors_u, errors_nu_seg


@benchmark
def error_onecore(nxvec, find_anlyt_error_func, setup_dict):
    global errors_u, errors_nu_seg
    errors_u, errors_nu_seg = initializer(nxvec)
    for indx, num_of_cells in enumerate(nxvec):
        merge_result_err(gen_errs(indx, num_of_cells, find_anlyt_error_func, setup_dict))
    return errors_u, errors_nu_seg



def calulate_all_errors(nxvec, find_anlyt_error_func, setup_dict):
    return error_onecore(nxvec, find_anlyt_error_func, setup_dict)


def  calulate_all_errors_bkup(nxvec, find_anlyt_error_func, setup_dict):
    # higher-order function
    import numpy as np

    errors_u = np.zeros(len(nxvec))
    errors_nu_seg = np.zeros(len(nxvec))
    for indx, num_of_cells in enumerate(nxvec):
        delta_space = gen_uniform_delta(num_of_cells)
        mesh =gen_xy_mesh(num_of_cells, delta_space)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing'] = delta_space
        final_phi = single_solve(setup_dict, mesh, None)  #False  True
        errors_u[indx], _ = find_anlyt_error_func(mesh, final_phi)
        
        delta_vect, delta_space = gen_nonuniform_segment_delta(num_of_cells)
        mesh =gen_xy_nnunif_mesh(delta_vect)
        setup_dict['grid_spacing'] = delta_space
        setup_dict['center_spacing']  = delta_space
        final_phi = single_solve(setup_dict, mesh, None)  #False  True
        errors_nu_seg[indx], _ = find_anlyt_error_func(mesh, final_phi)
        
    return  errors_u, errors_nu_seg




def plot_conv_acc(nxvec, errors_u, errors_nu_seg, setup_dict):
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
    specifc=setup_dict['eq_formula2']
    title_txt = f'{formula} \n {specifc}'
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
    file_name = '2_'+setup_dict['compactxt_eq']
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
    #fig.savefig(report_folder/(file_name+'.pdf'), bbox_inches="tight")
    fig.savefig(report_folder/(file_name+'.jpg'), bbox_inches="tight")
    #fig.savefig(report_folder/(file_name+'.png'), bbox_inches="tight")
    #fig.savefig(report_folder/(file_name+'.eps'), format='eps', dpi=300)
    #fig.savefig(report_folder/(file_name+'.svg'), format='svg', dpi=300)

    return fig
