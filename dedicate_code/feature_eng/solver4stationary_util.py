# -*- coding: utf-8 -*-
"""
Version 1, Generate and write solution
Genereate and write single (1) solution
Genereate and write single multiple solutions with same call in parallel
@author: enzo
"""

#%% Setup solver
import os
#from pickle import FALSE
#os.environ["FIPY_SOLVERS"] = "petsc" # petsc scipy no-pysparse trilinos  pysparse pyamg

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



#%% 1.  setups and solvers

def setup_phi_0(variables_dict, mesh):
    import numpy as np
    dim = variables_dict['dimensions']
    assert dim <4  and dim>0
    #phi = fi.CellVariable(name="phi", mesh=mesh, value=1.)
    #phi = fi.CellVariable(name="phi", mesh=mesh, value=0.)
    phi = fi.CellVariable(name="phi", mesh=mesh)

    #FixedFlux boundary condition is now deprecated.
    #https://www.ctcms.nist.gov/fipy/documentation/USAGE.html

    if variables_dict['boundaryConditions']=='Neumann':
        #fixed flux (Neumann condition)
        phi.faceGrad.constrain(0 * mesh.faceNormals, where=mesh.exteriorFaces)
    elif variables_dict['boundaryConditions']=='Dirichlet':
        phi.constrain(0., where=mesh.exteriorFaces)
    elif variables_dict['boundaryConditions']=='Dirichlet0.5':
        phi.constrain(0., where=mesh.facesLeft)
        phi.constrain(0.5, where=mesh.facesRight)
    elif variables_dict['boundaryConditions']=='None':
        pass    # il like Neumann
    else:
        bc = variables_dict['boundaryConditions']
        raise NameError(f'Not a valid boundaryCondition {bc}')
    return phi


def get_dirac(center, variables_dict, mesh):
    dim = variables_dict['dimensions']
    assert dim <4  and dim>0
    epsilon = variables_dict['epsilon_delta'] # grid_spacing
    delta_product = variables_dict['delta_product']
    #Inpulse generation
    dirac = fi.CellVariable(name="dirac", mesh=mesh,  value=0.)
    #TODO
    # logger.debug(f'1.9*delta*2= {1.4*(delta_space*2)} <? {9.9*10**-3}')
    appx_type = variables_dict['delta_type']
    dirac[:] = delta_func(mesh, center, epsilon=epsilon,\
             dim=dim, approx_type= appx_type, delta_product = delta_product)
    return dirac


def get_equation(variables_dict, mesh, flatten_field, phi):
    fpnp = fi.numerix
     

    source = variables_dict['source']
    F= fi.CellVariable(mesh=mesh, rank=0)
    if source == 'delta':
        F[:] =  get_dirac(0.0, variables_dict, mesh).value #- (1-x) 
        sourceCoeff = F
    elif source == 'delta0.5':
        F[:] =  get_dirac(0.5, variables_dict, mesh).value #- (1-x) 
        sourceCoeff = F
    elif source == 'Fdelta':
        F[:] =  flatten_field #- (1-x) 
        sourceCoeff = F  * get_dirac(0.0, variables_dict, mesh).value
    elif source == 'eFdelta':
        F[:] =  flatten_field #- (1-x) 
        sourceCoeff = fpnp.exp(-F) * get_dirac(0.0, variables_dict, mesh).value
    else:
        raise NameError('Not a valid source type')


    #%% Dif
    D = variables_dict['D']
    if D == 0.0:
        logger.debug(f'Alternative D!!!')
        D=fi.CellVariable(mesh=mesh, rank=0)
        D[:] =  fpnp.exp(flatten_field)

    #%% Equation
    convect = variables_dict['convection']
    U= fi.CellVariable(mesh=mesh, rank=0)
    U[:] =  flatten_field #- (1-x)
    #TODO gradient of U
    #grad_U =  U.faceGrad  #faceGrad  grad
    if convect:
        equ =  (fi.DiffusionTerm(coeff=D, var=phi)
                + fi.ExponentialConvectionTerm(coeff=U.grad, var=phi)
                - sourceCoeff)
    elif not convect:
        equ =  (fi.ImplicitDiffusionTerm(coeff=D, var=phi) # 
               == sourceCoeff)   # == 0.0
    else:
        logger.error(f'Not contemplated setup convect:{convect}')
        raise NameError(f'Not contemplated setup convect:{convect}')

    return equ

@benchmark
def swipe_solve(variables_dict, phi, equ, amgx_dict_name): #TODO add amgx_dict to variables_dict
    dim = variables_dict['dimensions']
    solver_type = variables_dict['solver_type']

    import numpy as np
    if dim == 1:
        iters = 4000    # default: 2000
        tol = 2.e-7    # default: 1.e-15
        desiredResidual =  2.e-5 # 5e-2  1e-5  1e-9 4e-1
    elif dim == 2:
        logger.debug(f'On 2D settings')
        iters = 4000 #9000
        tol = 2.e-7  #2.e-7
        desiredResidual = 2.e-5 # 1.e-12  4.e-3  # 4e-5  1e-5 1e-9 
    elif dim == 3:
        logger.debug(f'On 3D settings')
        iters = 7000  # 100 300
        tol = 5.e-4  # 4e-1  5e-2    5.e-2 4.e-2  4.e-3
        desiredResidual =  5.e-2   #4e-1  5e-2   4.e-3
    else:
        logger.error(f'Not valid dim:{dim}')
        raise NameError(f'Not valid dim:{dim}')


    if solver_type == 'AmgX':
        from dedicate_code.feature_eng.pyamgx_solver import PyAMGXSolver
        import json
        if 'precon' in variables_dict:
            amgx_dict_name = variables_dict['precon']
        from dedicate_code.config import data_dir
        file_conf = amgx_dict_name+'.json'
        f = open(f"{data_dir/'configs'/file_conf}")
        amg_cfg_dict = json.load(f)
        amg_cfg_dict['solver']['max_iters'] = iters
        amg_cfg_dict['solver']['tolerance'] = tol
        amg_cfg_dict['solver']['print_solve_stats'] = 1
        mySolver = PyAMGXSolver(amg_cfg_dict) #solver
        logger.debug(f"solver: {amg_cfg_dict['solver']['solver']}")
    elif solver_type.startswith('sci'):
        if solver_type.endswith('PCG'):
            mySolver = fi.LinearPCGSolver(iterations=3000, tolerance=2e-7) 
        elif solver_type.endswith('CGS'):
            mySolver = fi.LinearCGSSolver(iterations=1234, tolerance=2e-7) 
        elif solver_type.endswith('GMRE'):
            mySolver = fi.LinearGMRESSolver(iterations=1234, tolerance=2e-7) 
        elif solver_type.endswith('LU'):
            mySolver = fi.LinearLUSolver(iterations=1234, tolerance=2e-7) 
        else:
            logger.error(f'Not valid solver_type:{solver_type}')
            raise NameError(f'Not valid solver_type:{solver_type}')    
        import os
        logger.debug(f"solver - scipy type GMRES: {os.environ['FIPY_SOLVERS']}")
    elif solver_type.startswith('pet'):
        prec = variables_dict['precon']
        if solver_type.endswith('PCG'):
            mySolver = fi.LinearPCGSolver(iterations=4000, tolerance=2e-7, precon=prec) #1d:1024 2e-7,  2d: 4000 2e-5,
        elif solver_type.endswith('CGS'):
            mySolver = fi.LinearCGSSolver(iterations=1234, tolerance=2e-7, precon=prec) 
        elif solver_type.endswith('GMRE'):
            mySolver = fi.LinearGMRESSolver(iterations=1234, tolerance=2e-7, precon=prec) 
        elif solver_type.endswith('LU'):
            mySolver = fi.LinearLUSolver(iterations=1234, tolerance=2e-7, precon=prec) 
        else:
            logger.error(f'Not valid solver_type:{solver_type}')
            raise NameError(f'Not valid solver_type:{solver_type}')    
        import os
        logger.debug(f"solver - scipy type GMRES: {os.environ['FIPY_SOLVERS']}")




    else:
        logger.error(f'Not valid solver_type:{solver_type}')
        raise NameError(f'Not valid solver_type:{solver_type}')
    #tolerance=  1.e-7 1.e-6 5e-6    4.e-2
    #mySolver = fi.LinearGMRESSolver(iterations=iters, tolerance=tol) 
    #BEST FOR pyamg:
    #LinearGeneralSolver
    # precon only petsc
    #mySolver = fi.LinearPCGSolver(iterations=1234, tolerance=5e-6, precon='sor') 
    #mySolver = fi.LinearCGSSolver(iterations=1000,tolerance=1.e-7, precon='jacobi')
    #mySolver = fi.LinearGMRESSolver(iterations=1234,tolerance=5e-6, precon='ilu')
    #mySolver = fi.LinearLUSolver(iterations=1000,tolerance=1.e-7, precon='sor')
    #https://stackoverflow.com/questions/54634268/solver-tolerance-and-residual-error-when-using-sweep-function-in-fipy
    #https://fipy.nist.narkive.com/wIKBXdLD/residuals-in-fipy

    
    
    tryMax=5
    phi_tt = np.zeros(len(phi.value))

    residual=10
    try_count=0
    while residual> desiredResidual and try_count < tryMax:
        residual = equ.sweep(var=phi, solver=mySolver) #
        try_count +=1
        if solver_type == 'AmgX':
            stas = mySolver.solver.status
            iterations = mySolver.solver.iterations_number
            if stas.startswith('diverged') or (iterations >= (iters -2)):
                logger.critical(f'status: {stas} - {amgx_dict_name}')
                logger.critical(f'iternum: {iterations}')
            else:
                logger.debug(f'status: {stas} - {amgx_dict_name}')
                logger.debug(f'iternum: {iterations}')
            #print(f'residual:{residual}')
        logger.debug(f'swipe {try_count}, with residual {residual} ')
    if try_count >2 : 
        logger.info(f'Exceeded at residual {residual:.7f} with try_count {try_count}')



    if solver_type == 'AmgX':
        #print(f'tol: {mySolver.tolerance} ; inter {mySolver.iterations} ')
        #print(f'status: {mySolver.solver.status}')
        #iteration = mySolver.solver.iterations_number
        #print(f'iternum: {iteration}')
        #print(f'residual: {mySolver.solver.get_residual}')
        mySolver.destroy()

    phi_tt= phi.value.copy()
    del phi
    return phi_tt


#%% 1.  wrapper
from dedicate_code.feature_eng.pyamgx_solver import cfg_pcgJ_dict #cfg_dict

@benchmark
def single_solve(variables_dict, mesh, flatten_field, amgx_dict='AMG_CLASSICAL_PMIS'):
    """
    full time steps solver
    pde_moive_function
    """
    #%% 2-3.  Set initial conditions and boundary conditions
    phi = setup_phi_0(variables_dict, mesh)

    #%% 4-5.  set Param AND equation
    equ = get_equation(variables_dict, mesh, flatten_field, phi)

    logger.debug(f'Created equation')

    #%% 7. Solving PDE
    phi_tt  =swipe_solve(variables_dict, phi, equ, amgx_dict)
    logger.debug(f'Solved equation')
    del phi
    del equ
    
    return phi_tt

@benchmark
def solve(i, variables_dict, mesh, flatten_field, verbose=False): #TODO verbose
    """
    3 time steps output solver, for multi solutions merging
    """
    #%% 2-3.  Set initial conditions and boundary conditions
    phi = setup_phi_0(variables_dict, mesh)

    #%% 4-5.  set Param AND equation
    equ = get_equation(variables_dict, mesh, flatten_field, phi)
    logger.debug(f'Created equation')

    #%% 7. Solving PDE
    import numpy as np
    dim = variables_dict['dimensions']
    phi_tt = np.zeros((1+1, len(phi.value)))
    phi_tt[0]  = phi.value.copy()

    phi_tt[1] = swipe_solve(variables_dict, phi, equ, cfg_dict)

    if not i%50: logger.info(f'Step ***{i}***')
    if not i%10: logger.debug(f'Step ***{i}***')
       
    return (i, phi_tt)#, phi)


# %%
#%% 1.  Solutions generating with multi processor!

def initializer(mesh_len, number_of_fileds):
    import numpy as np
    sizes = ()
    logger.info(f'sizes={sizes}')
    sizes = (number_of_fileds,1+1, mesh_len)  #(len(fields), 3+1, size_mesh_x)
    logger.info(f'sizes={sizes}')
    solutions_array = np.zeros(sizes)#3D
    logger.info(f'fieldsShape={solutions_array.shape}' )
    return solutions_array

def init_pool(solutns):
    global solutions_global
    solutions_global = solutns

def get_results(result, start_ensamble):
    global solutions_global
    if not result[0]%5: 
        logger.debug(f'merging solutions; working on {result[0]-start_ensamble} i.e. {result[0]}-{start_ensamble}')
    solutions_global[result[0]-start_ensamble] = result[1]

@benchmark
def multi(variables_dict, mesh, fields_array, start_field=0):
    import multiprocessing as mp
    from functools import partial
    global solutions_global
    dims = variables_dict['dimensions']
    n_of_fileds = len(fields_array)
    logger.debug(f'multi run for  solutions')

    solutions_global = initializer(mesh.numberOfCells, n_of_fileds)
    pool = mp.Pool(mp.cpu_count()-1, init_pool(solutions_global))
    new_callback_function = partial(get_results, start_ensamble=start_field)
    for i in range(start_field, n_of_fileds):
        flatten_field = fields_array[i].flatten()
        pool.apply_async(solve, 
        args=(i, variables_dict, mesh, flatten_field), 
        callback=new_callback_function)
    pool.close()
    pool.join()
    return solutions_global

@benchmark
def single(variables_dict, mesh, fields_array, start_field=0):
    global solutions_global
    dims = variables_dict['dimensions']
    n_of_fileds = len(fields_array)

    logger.debug(f'single run for  solutions')
    solutions_global = initializer(mesh.numberOfCells, n_of_fileds)
    for i in range(start_field, n_of_fileds):
        flatten_field = fields_array[i].flatten()
        get_results(solve(i, variables_dict, mesh, flatten_field), start_field)
        del flatten_field
    return solutions_global



def ensamble_solutions(variables_dict, input_mesh, fields_array): #0.25 0.01   correlation length vector (m)
    #input_mesh = [x, y] is a list of the same lenght as dims
    #fields_array = fields_dict['fields_array']
    #input_mesh = fields_dict['mesh']
    dims = variables_dict['dimensions']
    #d_vector_mesh = array_from_mesh(dims, input_mesh)
    assert input_mesh.dim == dims

    logger.debug(f'generate solutions')

    #rng = range(start_ensamble, start_ensamble+ens_no)
    #all_sols = multi(input_mesh, dims, start_ensamble, ens_no)
    all_sols = single(variables_dict, input_mesh, fields_array)
    return_dict={}
    return_dict['final_solutions_array'] = all_sols[:,1]
    #return_dict['starting_guess_array'] = all_sols[:,0] #all zeros
    assert len(all_sols[:,1]) == len(all_sols[:,0])
    #assert len(return_dict['final_solutions_array']) == len(return_dict['starting_guess_array'])
    return return_dict