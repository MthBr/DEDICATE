#%% Setup solver
import os
os.environ["FIPY_SOLVERS"] = "scipy" #pyamgx petsc scipy no-pysparse trilinos  pysparse pyamg

os.environ["FIPY_VERBOSE_SOLVER"] = "1" # 1:True # Only for TESTING
#print('Verbose ' + os.environ.get('FIPY_VERBOSE_SOLVER'))
os.environ["OMP_NUM_THREADS"]= "1"


#%% Setup parameters

_dim_set = 2000
_test_set_cofig = 500

_mesh_config= 5


#Defaults! Nones
id= f'Test_00' #TODO unused#
dxyz = None #TODO remove
mc_runs = 1 #TODO delte

delta_type = '2*-cos' # l  cos Cubic LL  3*-f_arcs  # 2-l  1-l  1*-l 2-cos 2-Cubic 2-LL
coeff_delta_mesh= 1.5
epsilon_delta = 1.0
tensor_product = True



if _mesh_config==5:
    grid_type='uniform' #TODO add
    nxyz = 512 #256 #126  #512 1024
    center = 0
    leng = 2
    start = -1
if _mesh_config==1:
    dxyz = None #TODO remove
    nxyz = 1024 #256 #126  #512 1024
    center = 0
    leng = 2
    start = -1


if _dim_set == 1000:
    dims = 1
if _dim_set == 2000: #TODO 2000
    dims = 2
if _dim_set == 3000:
    dims = 3



if _test_set_cofig==500:
    static_prob = True
    BCs = 'Dirichlet' # Dirichlet Neumann None
    dims = 2
    _dim_set == 2000

    D = 1.0
    convect = False
    phi_0 = 'None'
    source = 'delta' # delta Fdelta  eFdelta 

    eq_formula = fr'$ \nabla \cdot ({D} \nabla h) = h_{{xx}} =  \delta^{{{delta_type}}}(x-0) , x\in[-1,1]^{dims}\subset R^{dims}$'
    eq_formula2 = rf'{BCs}=0,  SOL: $h(x)=   \frac{{ln(x/l_0)}}{{2 \pi }} , Norm:2     l_0=1 $'

if _test_set_cofig==700:
    static_prob = True
    dims = 3
    _dim_set == 3000
    BCs = 'Dirichlet' # Dirichlet Neumann None

    D = 1.0
    convect = False
    phi_0 = 'None'
    source = 'delta' # delta Fdelta  eFdelta 
    
    eq_formula = fr'$ \nabla \cdot ({D} \nabla h) = h_{{xx}} =  \delta^{{{delta_type}}}(x-0) , x\in[-1,1]^{dims}\subset R^{dims}$'
    eq_formula2 = rf'{BCs}=0,  SOL: $h(x)=  - \frac{{1}}{{4 \pi x {D} }} , Norm:2     l_0=1 $'



id=  f'Test_{_test_set_cofig+_dim_set}'

# controls
import math
#assert isinstance(correl, float)
assert isinstance(mc_runs, int)
assert isinstance(dims, int)
assert isinstance(D, float)
assert isinstance(convect, bool)
#assert isinstance(is_log_normal, bool)
assert isinstance(eq_formula, str)
#assert isinstance(text_long, str) #eq_formula2
#assert isinstance(compact_text, str)


#TODO revisit dxyz
#if dxyz!= None: 
#    assert math.isclose(dxyz*nxyz,1.0, rel_tol=10**-3)

# assign
setup_dict = {}

#setup_dict['grid_spacing'] = dxyz
#setup_dict['center_spacing'] = dxyz #used for delta Dirac
setup_dict['id'] = id
setup_dict['number_of_cells'] = nxyz
setup_dict['length'] = leng
setup_dict['start'] = start



setup_dict['dimensions'] = dims

setup_dict['D'] = D
setup_dict['convection'] = convect
setup_dict['initial'] = phi_0

setup_dict['delta_type'] = delta_type
setup_dict['epsilon_delta'] = epsilon_delta
setup_dict['tensor_product'] = tensor_product


setup_dict['source'] = source 
setup_dict['boundaryConditions'] = BCs 

setup_dict['eq_formula'] = eq_formula
setup_dict['eq_formula2'] = eq_formula2  ###eq_formula2    text_long


#setup_dict['compactxt_field'] =  f'cor{int(correl*1000)} variance {int(var*1000)}'  #
setup_dict['compactxt_mesh'] =  f'mesh{nxyz}'  #
setup_dict['compactxt_eq'] =  f'D{int(D*10)}_dlt:{delta_type}'  #
setup_dict['compactxt_solver'] = f'GMRE{os.environ["FIPY_SOLVERS"]}'  #TODO amgx....
setup_dict['compactxt_delta'] = f'dlt:{delta_type}_pnt{coeff_delta_mesh}'  #TODO amgx....




#setup_dict['filed_hash_name'] = f"dim{dims}_isLN{is_log_normal}_istr{start}_ilen{leng}_cor{int(correl*100)}_var{int(var*100)}_n{nxyz}"
setup_dict['sol_hash_name'] = f"{id}_{dims}d_{setup_dict['compactxt_eq']}_{setup_dict['compactxt_mesh']}"


setup_dict['hash_log_name'] = f'{id}_dim{dims}'
