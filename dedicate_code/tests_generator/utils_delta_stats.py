# -*- coding: utf-8 -*-
"""
Version 1, Exploration UTILS
Plots for showing fields or single solutions images and movies.
Plot by saving on file.
@author: enzo
"""
#%% import pakages
import matplotlib.pyplot as plt

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
mpl.rcParams["figure.titleweight"] = 'regular'
mpl.rcParams['legend.fontsize'] = legfsize
mpl.rcParams['lines.linewidth'] = lwidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = markeredgewidth


#%% define
def plot_nsave_solution(x_vect, analyt, numerc, title_str, file_name_dir):

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_vect, analyt, label="analytical")
    ax.plot(x_vect, numerc, linestyle="--", label="h")
    ax.legend(loc="best")
    ax.set_xlabel("x(0.5)")
    ax.set_ylabel("h")
    ax.set_title(title_str)
    fig.tight_layout()


    fig.savefig(file_name_dir, bbox_inches="tight")

    return fig


def plot_nsave_trend(nxvec, errors, title_str, file_name_dir):
    import numpy as np
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1)

    log_nxvec = np.log(nxvec)
    log_errors_u = np.log(errors)
    fit_u = np.polyfit(log_nxvec, log_errors_u, 1)

    ax.plot(log_nxvec, log_errors_u, label="simulation errors")
    ax.plot(log_nxvec, np.poly1d(fit_u)(log_nxvec), linestyle="--",
            label="Line: {m:1.2f}x+{b:1.2f}".format(m=fit_u[0], b=fit_u[1]))
    ax.legend(loc="best")
    ax.set_xlabel("log(nx)")
    ax.set_ylabel("log(error)")
    ax.set_title(title_str)

    ax = fig.add_subplot(1, 2, 2)

    fit_u = np.polyfit(nxvec, errors, 1)

    ax.plot(nxvec, errors, label="simulation errors")
    ax.plot(nxvec, np.poly1d(fit_u)(nxvec), linestyle="--",
            label="Line: {m:1.2f}x+{b:1.2f}".format(m=fit_u[0], b=fit_u[1]))
    ax.legend(loc="best")
    ax.set_xlabel("nx")
    ax.set_ylabel("error")
    ax.set_title(title_str)










    fig.savefig(file_name_dir, bbox_inches="tight")

    return fig

def gen_analyt_sol(setup_dict, mesh):
    dim = setup_dict['dimensions']
    import numpy as np
    import fipy as fp

    if dim == 2:
        fpnp = fp.numerix
        den = 2*fpnp.pi*setup_dict['D']
        analyt_sol = fp.CellVariable(name="analytical solution", mesh=mesh, value=0.)
        X = np.linalg.norm([mesh.x.value, mesh.y.value], ord=2, axis=0)
        l0=.5  #np.sqrt(1.5**2+1.5**2) #1.5
        analyt_sol[:] =  (fpnp.log(2*X) / den)

    elif dim ==1:
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

    return analyt_sol.value

def set_log_err(num_levels, errors, h_deltas, j, nxvec, save_to, delta_tp, ksup):
    from math import log as ln  # log is a fenics name too
    import numpy as np

    rates_array = np.zeros(num_levels)
    for i in range(1, num_levels):
        Ei = errors[i][j]
        Eim1 = errors[i - 1][j]
        r = ln(Ei / Eim1) / ln(h_deltas[i] / h_deltas[i - 1])
        rates_array[i] = round(r, 12)
    
    rate_leap = ln(errors[-1][j] / errors[0][j]) / ln(h_deltas[-1] / h_deltas[0])
    best_mean = np.mean(rates_array[3:])
    rates_array[0] = 0.0
    avg_err= np.format_float_positional(np.mean(errors[:,j]),9)


    #%%  Show convergence accuracy
    title_str = f'{delta_tp} lambda:{ksup} \n rel_err={best_mean:.5f},'
    placeholder_fig = plot_nsave_trend(nxvec, errors[:,j], title_str, save_to)
    placeholder_fig.clear()
    plt.close(placeholder_fig) 

    rates_array=np.hstack((np.format_float_positional(best_mean,5), rates_array))
    rates_array=np.hstack((rate_leap, rates_array))
    rates_array=np.hstack((avg_err, rates_array))
    
    rates_array=np.hstack((rates_array, errors[:,j]))
    return rates_array











