#%%


from fipy import CellVariable, Grid1D, DiffusionTerm, PowerLawConvectionTerm, ImplicitSourceTerm, Viewer
from fipy.tools import numerix

L = 10.
nx = 5000
dx =  L / nx
mesh = Grid1D(dx=dx, nx=nx)
phi0 = 1.0
alpha = 1.0
phi = CellVariable(name=r"$\phi$", mesh=mesh, value=phi0)
solution = CellVariable(name=r"solution", mesh=mesh, value=phi0 * numerix.exp(-alpha * mesh.cellCenters[0]))

#%%
from fipy import input
if __name__ == "__main__":
    viewer = Viewer(vars=(solution, phi))
    viewer.plot()
    input("press key to continue")

phi.constrain(phi0, mesh.facesLeft)

## fake outflow condition
phi.faceGrad.constrain([0], mesh.facesRight)
eq = PowerLawConvectionTerm((1,)) + ImplicitSourceTerm(alpha)
eq.solve(phi)
print(numerix.allclose(phi, phi0 * numerix.exp(-alpha * mesh.cellCenters[0]), atol=1e-3))

#%%
from fipy import input
if __name__ == "__main__":
    viewer = Viewer(vars=(solution, phi))
    viewer.plot()
    input("finished")





# %%
