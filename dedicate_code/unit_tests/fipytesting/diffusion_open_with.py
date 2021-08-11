
#%%
import os
os.environ["FIPY_SOLVERS"] = "scipy" #pyamgx petsc scipy no-pysparse trilinos  pysparse pyamg
os.environ["FIPY_VERBOSE_SOLVER"] = "1" # 1:True # Only for TESTING
os.environ["OMP_NUM_THREADS"]= "1"

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


from pyamgx_solver import PyAMGXSolver, cfg_dict


cfg_dict['solver']['max_iters'] = 1000
cfg_dict['solver']['print_solve_stats'] = 1
gmres_solver = PyAMGXSolver(cfg_dict) #solver
DiffusionTerm().solve(var=phi, solver=gmres_solver)
gmres_solver.destroy()

print(phi(((nx/2,), (nx/2))))
# %%
