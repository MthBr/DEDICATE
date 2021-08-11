# -*- coding: utf-8 -*-
"""
Version 1, Generate and write solution
Genereate and write single (1) solution
Genereate and write single multiple solutions with same call in parallel
@author: enzo
"""

#%% Setup solver
#import os
#os.environ["FIPY_SOLVERS"] = "scipy" # petsc scipy no-pysparse trilinos  pysparse pyamg

#os.environ["FIPY_VERBOSE_SOLVER"] = "1" # 1:True # Only for TESTING
#print('Verbose ' + os.environ.get('FIPY_VERBOSE_SOLVER'))
#os.environ["OMP_NUM_THREADS"]= "1"

#%% import pakages
import fipy as fi
from dedicate_code.feature_eng.delta_func_utils import delta_func

from dedicate_code.custom_funcs import  get_logger, benchmark
global logger
#logger = get_logger(log_dir/'solver_util.log', 'general')
logger = get_logger()


global time_offset 
time_offset = 12*10**-7



#%% 1.  setups and solvers

def setup_phi_0(variables_dict, mesh):
    import numpy as np
    dim = variables_dict['dimensions']
    assert dim <4  and dim>0

    delta_space = variables_dict['center_spacing']

    if variables_dict['initial'] == 'delta':
        appx_type = variables_dict['delta_type']
        phi = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
        logger.info(f'5*dleta*2= {1.4*(delta_space*2)} <? {9.9*10**-3}')
        #assert(5*(delta_space*moment) < 9.9*10**-3)
        #TODO review cneter!!! no more 0.5 but MID? mesh?
        phi[:] = delta_func(mesh, 0.5, h=1.4*delta_space,\
             dim=dim, approx_type= appx_type)

    elif variables_dict['initial'] == '50x(1-x)':
        fpnp = fi.numerix
        phi = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
        X = mesh.x.value
        phi[:] = 50.* X * (1.-X)
    elif variables_dict['initial'] == 'const3':
        #else phi inital is constant, equal to 3!!!
        phi = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=3.)
    else:
        inc = variables_dict['initial']
        raise NameError(f'Not a valid initial condition: {inc}')

    #FixedFlux boundary condition is now deprecated.
    #https://www.ctcms.nist.gov/fipy/documentation/USAGE.html

    if variables_dict['boundaryConditions']=='Neumann':
        #fixed flux (Neumann condition)
        phi.faceGrad.constrain(0 * mesh.faceNormals, where=mesh.exteriorFaces)
    elif variables_dict['boundaryConditions']=='Dirichlet':
        phi.constrain(0., where=mesh.exteriorFaces)
    elif variables_dict['boundaryConditions']=='None':
        pass    
    else:
        bc = variables_dict['boundaryConditions']
        raise NameError(f'Not a valid boundaryCondition {bc}')
    return phi


def get_dirac(center, variables_dict, mesh):
    dim = variables_dict['dimensions']
    assert dim <4  and dim>0
    delta_space = variables_dict['grid_spacing']
    #Inpulse generation
    dirac = fi.CellVariable(name="dirac", mesh=mesh, hasOld=True,  value=0.)
    logger.info(f'1.9*delta*2= {1.4*(delta_space*2)} <? {9.9*10**-3}')
    appx_type = variables_dict['delta_type']
    dirac[:] = delta_func(mesh, center, h=1.4*delta_space,\
             dim=dim, approx_type= appx_type)
    return dirac


def get_equation(variables_dict, mesh, flatten_field, phi):
    fpnp = fi.numerix
    D = variables_dict['D']   

    source = variables_dict['source']
    F= fi.CellVariable(mesh=mesh, rank=0)
    if source == 'None':
        doSource = False
        #sourceCoeff = F * 0.
    elif source == 'Fconst':
        doSource = True
        F[:] =  flatten_field #- (1-x) 
        sourceCoeff = F *10.
    elif source == 'Fdelta':
        doSource = True
        F[:] =  flatten_field #- (1-x) 
        sourceCoeff = F  * get_dirac(0.0, variables_dict, mesh).value
    elif source == 'e-F*delta':
        doSource = True
        F[:] =  flatten_field #- (1-x) 
        sourceCoeff = fpnp.exp(-F) * get_dirac(0.0, variables_dict, mesh).value
    else:
        raise NameError('Not a valid source type')

    #%% Equation
    convect = variables_dict['convection']
    U= fi.CellVariable(mesh=mesh, rank=1)
    U[:] =  flatten_field #- (1-x) 
    if convect and doSource:
        equ =  (fi.TransientTerm(var=phi) ==
                fi.DiffusionTerm(coeff=D, var=phi)
                + sourceCoeff
                - fi.ConvectionTerm(coeff=U, var=phi))
    elif convect and not doSource :
        equ =  (fi.TransientTerm(var=phi) ==
                fi.DiffusionTerm(coeff=D, var=phi)
                - fi.ConvectionTerm(coeff=U, var=phi))
    elif not convect and doSource:
        equ =  (fi.TransientTerm(var=phi) ==
                fi.DiffusionTerm(coeff=D, var=phi)
                + sourceCoeff)
    elif not convect and not doSource:
        equ =  (fi.TransientTerm(var=phi) ==
                fi.DiffusionTerm(coeff=D, var=phi))
    else:
        logger.error(f'Not contemplated setup convect:{convect} and source:{doSource}')
        raise NameError(f'Not contemplated setup convect:{convect} and source:{doSource}')

    return equ


def get_dt(variables_dict, flatten_field):
    import numpy as np
    dim = variables_dict['dimensions']
    delta_space = variables_dict['grid_spacing']

    D = variables_dict['D'] 

    convect = variables_dict['convection'] #U
    time_cfl = 1
    U_max = 1

    #Von Neumann stability 
    #https://ocw.mit.edu/courses/mathematics/18-336-numerical-methods-for-partial-differential-equations-spring-2009/lecture-notes/MIT18_336S09_lec14.pdf
    safetyFactor = 0.9
    time_vns = safetyFactor * delta_space**2 / (2 * dim * D)

    # CourantSafetyNumber
    # This is the maximum fraction of the CFL-implied timestep that will be usedto advance any grid.
    # A value greater than 1 is unstable (for all explicit methods).  
    # The recommended value is0.4. Default: 0.6
    # For some schemes used for incompressible flows you might need c0 < 0.5. 
    CourantSafetyNumber = 0.45
    if convect: 
        U_max = np.linalg.norm(flatten_field, np.inf)
        time_cfl = CourantSafetyNumber/dim * delta_space/U_max
    
    logger.info(f'D:{D}, dim:{dim}, Umax:{U_max}')
    logger.info(f'time_cfl:{time_cfl}, time_vns:{time_vns}')
    delta_t = min(time_cfl, time_vns)

    if delta_t > time_offset :
        logger.warning(f'delta_t = {delta_t} > {time_offset} = time_offset')
    else:
        logger.error(f'delta_t = {delta_t} < {time_offset} = time_offset')
    #assert(delta_t > time_offset)
    

    clf = dim * U_max* delta_t / delta_space
    vns = dim *D*delta_t/ delta_space**2
    logger.warning(f'clf: {clf} < 0.5; vns: {vns} < 0.5')
    if convect: assert(clf < 0.5)
    assert(vns < 0.5)
    return delta_t, vns, clf 


def adjust_dt(totalTime, delta_t):
    logger.debug(f'Adjusting time! totalTime = {totalTime}; delta_t{delta_t}')
    delta_t=totalTime/3 + 0.5*totalTime/3
    steps = int(totalTime/delta_t)
    logger.debug(f'Now: steps = {steps}; delta_t{delta_t}')
    return steps, delta_t


def swipe_solve(phi, equ, delta_t, totalTime, verbose, saveEach = False, vi=None):
    import numpy as np
    #tolerance=  1.e-7 1.e-6 5e-6
    mySolver = fi.LinearCGSSolver(iterations=1234, tolerance=5e-6) 
    # precon only petsc
    #mySolver = fi.LinearPCGSolver(iterations=1234, tolerance=5e-6, precon='sor') 
    #mySolver = fi.LinearCGSSolver(iterations=1000,tolerance=1.e-7, precon='jacobi')
    #mySolver = fi.LinearGMRESSolver(iterations=1234,tolerance=5e-6, precon='ilu')
    #mySolver = fi.LinearLUSolver(iterations=1000,tolerance=1.e-7, precon='sor')
    #https://stackoverflow.com/questions/54634268/solver-tolerance-and-residual-error-when-using-sweep-function-in-fipy
    #https://fipy.nist.narkive.com/wIKBXdLD/residuals-in-fipy
    desiredResidual = 5e-2 #  1  1e-5  1e-9
    elapsedTime = 0
    tryMax=10

    steps = int(totalTime/delta_t)
    if steps == 0: 
        steps,delta_t = adjust_dt(totalTime, delta_t)

    import math
    if  math.isclose(delta_t*steps,totalTime, rel_tol=10**-16):  # delta_t*steps == totalTime
        logger.debug(f'N steps delta_t*steps {delta_t*steps} = {totalTime} totalTime')
        total_steps=steps+1
        estra_step=False
    elif delta_t*steps < totalTime:
        logger.debug(f'N+1 steps delta_t*steps {delta_t*steps} < {totalTime} totalTime')
        total_steps=steps+2
        estra_step=True
    else:
        logger.critical(f'delta_t*steps {delta_t*steps} > {totalTime} totalTime')
        raise NameError(f'delta_t*steps {delta_t*steps} > {totalTime} totalTime')


    if saveEach: 
        phi_tt = np.zeros((total_steps, len(phi.value)))
        phi_tt[0]  = phi.value.copy()
    else:
        phi_tt = np.zeros(len(phi.value))
    logger.debug(f'N steps {steps}')

    for step in range(steps):
        phi.updateOld()
        residual=10
        try_count=0
        while residual> desiredResidual and try_count < tryMax:
            residual = equ.sweep(var=phi, dt=delta_t, solver=mySolver) #
            try_count +=1
        elapsedTime += delta_t
        #if not step%100: print(step)
        if try_count >2 : logger.debug(f'Exceeded at step {step} with try_count{try_count}')
        if saveEach & verbose: 
            vi.plot()
            print(f'time_step = {step} over {steps}')
        if saveEach: 
            phi_tt[step+1] = phi.value.copy()

        if not step%300: logger.debug(f'at step = {step} over {steps}')


    logger.debug(f'final step = {step} over {steps}, size phi_tt:{len(phi_tt)}')
    logger.debug(f'elapsedTime = {elapsedTime}; totalTime {totalTime}')
    

    final_timeStep = totalTime-elapsedTime
    if final_timeStep - delta_t < 2*10**-16:
        logger.debug(f'final_timeStep = {final_timeStep} < delta_t {delta_t}')
    else:
        logger.critical(f'final_timeStep = {final_timeStep} >= delta_t {delta_t}')
    #assert(final_timeStep - delta_t < 2*10**-16) # APPROX final_timeStep < delta_t
    #math.isclose(final_timeStep,delta_t, rel_tol=10**-12)

    if estra_step:
        #assert(final_timeStep - delta_t < 2*10**-16) # APPROX final_timeStep < delta_t
        assert(final_timeStep > 0)
        phi.updateOld()
        residual=10
        try_count=0
        while residual> desiredResidual and try_count < tryMax:
            residual = equ.sweep(var=phi, dt=final_timeStep, solver=mySolver) #
            try_count +=1
        elapsedTime += final_timeStep
        assert(elapsedTime == totalTime)

        if saveEach:
            phi_tt[step+2]= phi.value.copy() # phi_tt[steps+1]= phi.value.copy()

    
    assert(math.isclose(elapsedTime,totalTime, rel_tol=10**-12))
    #TODO would like rel_tol=10**-16
    if not math.isclose(elapsedTime,totalTime, rel_tol=10**-16):
        logger.warning(f'elapsedTime = {elapsedTime} NEQ totalTime {totalTime}; DIFF {totalTime- elapsedTime}')
    #Note;  ForSTEPS[elapsedTime += delta_t]  IS DIFFERENT FROM   delta_t*steps

    if not saveEach:
        phi_tt= phi.value.copy()
    
    return phi_tt, phi



#%% 1.  wrapper

@benchmark
def single_movie_solve(variables_dict, mesh, flatten_field, timeDuration, verbose=False):
    """
    full time steps solver
    pde_moive_function
    """
    #TODO verify time_offset!
    #timeDuration = timeDuration - time_offset

    #file_name=variables_dict['compactxt']+'.log'
    #logger = get_logger(log_dir/file_name, 'specific', False)


    #%% 2-3.  Set initial conditions and boundary conditions
    phi = setup_phi_0(variables_dict, mesh)

    #%% 4-5.  set Param AND equation
    equ = get_equation(variables_dict, mesh, flatten_field, phi)

    #%% Time setup - stability
    delta_t, vns, clf = get_dt(variables_dict, flatten_field)
    logger.info(f'delta_t = {delta_t}; VonNeumannStab:{vns}, CouranrNum:{clf:.5f}')
    
    if verbose:
         vi = fi.Viewer((phi))
    else:
        vi = None

    #%% 7. Solving PDE
    phi_tt, _  =swipe_solve(phi, equ, delta_t, timeDuration, verbose, saveEach = True, vi=vi)

    return phi_tt


@benchmark
def de_f_time(i, variables_dict, mesh, flatten_field, timeDuration, verbose=False):
    """
    3 time steps output solver, for multi solutions merging
    """
    #TODO verify time_offset!
    #timeDuration = timeDuration - time_offset
    #%% 2-3.  Set initial conditions and boundary conditions
    phi = setup_phi_0(variables_dict, mesh)

    #%% 4-5.  set Param AND equation
    equ = get_equation(variables_dict, mesh, flatten_field, phi)

    #%% Time setup - stability
    delta_t, vns, clf = get_dt(variables_dict, flatten_field)

    if not i%20:
        print(f'{i}, VonNeumann={vns:.5f}, CouranrNum={clf:.5f} ')
        print(f'{i}, dtnew = {delta_t}')
    
    logger.debug(f'{i}, VonNeumann={vns:.5f}, CouranrNum={clf:.5f}, dtnew = {delta_t} ')

    #%% 7. Solving PDE
    import numpy as np
    phi_tt = np.zeros((3+1, len(phi.value)))
    phi_tt[0]  = phi.value.copy()

    timeDuration1 = timeDuration/10
    if not i%20:
        logger.info(f'{i}, FIST integration d_time={timeDuration1}')
    else:
        logger.debug(f'{i}, FIST integration d_time={timeDuration1}')
    phi_tt[1], phi = swipe_solve(phi, equ, delta_t, timeDuration1, verbose)


    timeDuration2 = timeDuration/2 - timeDuration1
    if not i%20:
        logger.info(f'{i}, MID integration d_time={timeDuration2}')
    else:
        logger.debug(f'{i}, MID integration d_time={timeDuration2}')
    phi_tt[2], phi = swipe_solve(phi, equ, delta_t, timeDuration2, verbose)


    if not i%20:
        logger.info(f'{i}, VonNeumann={vns:.5f}, CouranrNum={clf:.5f}, dtnew = {delta_t}')
        logger.info(f'{i}, minstep={timeDuration1}, mildStep={timeDuration2}, midStep={timeDuration/2}')
    else:
        logger.debug(f'{i}, VonNeumann={vns:.5f}, CouranrNum={clf:.5f}, dtnew = {delta_t}')
        logger.debug(f'{i}, minstep={timeDuration1}, mildStep={timeDuration2}, midStep={timeDuration/2}')
       
    
    phi_tt[3], phi = swipe_solve(phi, equ, delta_t, timeDuration/2, verbose)


    return (i, phi_tt, delta_t)


# %%

# %%
