# Solve the steady-state convection-diffusion equation in one dimension.
#https://www.ctcms.nist.gov/fipy/examples/convection/generated/examples.convection.exponential1D.mesh1D.html



#%%
diffCoeff = 1.
convCoeff = (10.,)

#We define a 1D mesh
from fipy import CellVariable, Grid1D, DiffusionTerm,ConvectionTerm, PowerLawConvectionTerm, ExponentialConvectionTerm, Viewer
from fipy.tools import numerix
L = 10.
nx = 10
mesh = Grid1D(dx=L / nx, nx=nx)
valueLeft = 0.
valueRight = 1.

#The solution variable is initialized to valueLeft:
var = CellVariable(mesh=mesh, name="variable")
#and impose the boundary conditions with
var.constrain(valueLeft, mesh.facesLeft)
var.constrain(valueRight, mesh.facesRight)

#The equation is created with the DiffusionTerm and 
# ExponentialConvectionTerm. 
# The scheme used by the convection term needs to calculate a 
# Péclet number and thus the diffusion term instance must 
# be passed to the convection term.

eq = (DiffusionTerm(coeff=diffCoeff)
      + ExponentialConvectionTerm(coeff=convCoeff))

#We solve the equation

eq.solve(var=var)
#and test the solution against the analytical result

#%%
axis = 0
x = mesh.cellCenters[axis]
CC = 1. - numerix.exp(-convCoeff[axis] * x / diffCoeff)
DD = 1. - numerix.exp(-convCoeff[axis] * L / diffCoeff)
analyticalArray = CC / DD
print(var.allclose(analyticalArray))

#If the problem is run interactively, we can view the result:


#%%
if __name__ == '__main__':
    viewer = Viewer(vars=var)
    viewer.plot()

# %%
