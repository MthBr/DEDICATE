
#%%
import os
os.environ["AMGX_DIR"] = "/home/modal/Projets_local/AMGX-2.2.0" #pyamgx petsc scipy no-pysparse trilinos  pysparse pyamg

#%%
from fipy import Grid2D, CellVariable, DiffusionTerm

nx = 1000
ny = nx
dx = 1.
dy = dx
L = dx * nx
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.)

D = 1.

valueTopLeft = 0
valueBottomRight = 1

X, Y = mesh.faceCenters
facesTopLeft = ((mesh.facesLeft & (Y > L / 2))
                 | (mesh.facesTop & (X < L / 2)))
facesBottomRight = ((mesh.facesRight & (Y < L / 2))
                     | (mesh.facesBottom & (X > L / 2)))

phi.constrain(valueTopLeft, facesTopLeft)
phi.constrain(valueBottomRight, facesBottomRight)

import time

"""
t1 = time.time()
DiffusionTerm().solve(var=phi)
t2 = time.time()
"""
import pyamgx
import json


pyamgx.initialize()

from pyamgx_solver import PyAMGXSolver
with open (os.environ['AMGX_DIR']+'/core/configs/AMG_CLASSICAL_PMIS.json') as f:
    cfg = json.load(f)

cfg['solver']['max_iters'] = 1000
cfg['solver']['print_solve_stats'] = 1
solver = PyAMGXSolver(cfg)
gmres = PyAMGXSolver(cfg_dict)

t1 = time.time()
DiffusionTerm().solve(var=phi, solver=solver)
t2 = time.time()
solver.destroy()
pyamgx.finalize()

print(phi(((nx/2,), (nx/2))))
print("Time for solve: {} seconds.".format(t2-t1))
# %%
