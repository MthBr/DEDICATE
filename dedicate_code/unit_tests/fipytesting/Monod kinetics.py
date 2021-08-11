
#%%

from fipy import CellVariable, Grid1D,Viewer, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix

size, dl, q = 100, 3e-6, 14.1
Ds, Ks = 1e-4, 5.8e-4
order = 2
dt = dl ** 2 / (2 * Ds)

mesh = Grid1D(dx=dl, nx=size)
phi_s = CellVariable(name='substrate', mesh=mesh)
B = CellVariable(name='biomass', mesh=mesh)

B.value[-10:] = 1000
phi_s.constrain(1, mesh.facesLeft)
#reactions = - q * B * (phi_s / (phi_s + Ks))
reactionsCoeff = - q * B * (phi_s + Ks)

phi_s.setValue(numerix.linspace(1, 0, size))
#equation = DiffusionTerm(coeff=Ds) + reactions == 0
equation = DiffusionTerm(coeff=Ds) + ImplicitSourceTerm(coeff=reactionsCoeff) == 0

res = 1e8
while res > 1e-4:
    res = equation.sweep(var=phi_s)
    print(phi_s.value[::20])



# %%
