# -*- coding: utf-8 -*-
"""
Version 1, Extraction UTILS
generete mesh
@author: enzo
"""
#%% import pakages
import fipy as fi

from dedicate_code.custom_funcs import  get_logger, benchmark
global logger
logger = get_logger()


#%% 1.  create a mesh
def order_mesh_old(seq, size):
    import numpy as np
    seen = set()
    seen_add = seen.add
    list_out =  [x for x in seq if not (x in seen or seen_add(x))]
    assert len(list_out) ==size
    return np.array(list_out)

def reorder_mesh(seq, size):
    import numpy as np
    array_out = np.unique(seq)
    assert len(array_out) ==size
    return array_out



def array_from_mesh(dim, mesh, lens=None):
    array = []
    if lens == None:
        nx = mesh.nx
        ny = mesh.nx #TODO
        nz = mesh.nx #TODO
    else:
        nx = lens
        ny = lens
        nz = lens

    if dim==1:
        array = [mesh.x.value]
    elif dim == 2:
        xy = mesh.cellCenters.value
        x = reorder_mesh(xy[0],nx)
        y = reorder_mesh(xy[1],ny)
        array = [x,y]
    elif dim ==3:
        xyz = mesh.cellCenters.value
        x = reorder_mesh(xyz[0], nx)
        y = reorder_mesh(xyz[1], ny)
        z = reorder_mesh(xyz[2], nz)
        array = [x,y,z]
    return array






def gen_x_mesh(nx, dx, start = 0.0): #101   # 0.1
    mesh = fi.Grid1D(nx=nx, dx= dx)
    mesh = (mesh + start)
    #x = mesh.cellCenters[0].value  #x = mesh.x.value
    return mesh#, [x]

def gen_x_nnunif_mesh(dxvec, start = 0.0):
    mesh = fi.Grid1D(dx= dxvec)
    mesh = (mesh + start)
    #x = mesh.cellCenters[0].value
    return mesh#, [x]


def gen_xy_mesh(nxy, dxy, start = [[0.0],[0.0]]): #nxy = 101 , dxy =.01
    mesh = fi.Grid2D(nx=nxy, dx= dxy, ny=nxy, dy= dxy)
    mesh = (mesh + start)
    #xy = mesh.cellCenters.value
    #x = order_mesh(xy[0], mesh.nx)
    #y = order_mesh(xy[1], mesh.ny)
    return mesh#, [x, y]

def gen_xy_nnunif_mesh(dxvec, start = [[0.0],[0.0]]):
    mesh = fi.Grid2D(dx= dxvec, dy= dxvec)
    mesh = (mesh + start)
    # xy = mesh.cellCenters.value
    # x = order_mesh(xy[0], len(dxvec))
    # y = order_mesh(xy[1], len(dxvec))
    return mesh#, [x, y]

def gen_xyz_mesh(nxyz , dxyz, start = [[-1.0],[-1.0],[-1.0]]): #nxyz = 101 , dxyz =.01
    logger.debug(f'generating 3D mesh...')
    mesh = fi.Grid3D(dx = dxyz, dy = dxyz, dz = dxyz, \
        nx = nxyz, ny = nxyz, nz = nxyz)
    mesh = (mesh + start)
    logger.debug(f'generated 3D mesh...')
    # xyz = mesh.cellCenters.value
    # x = order_mesh(xyz[0], mesh.nx)
    # y = order_mesh(xyz[1], mesh.ny)
    # z = order_mesh(xyz[2], mesh.nz)
    return mesh#, [x, y, z]

def gen_xyz_nnunif_mesh(dxvec, start = [[-1.0],[-1.0],[-1.0]]):
    mesh = fi.Grid3D(dx = dxvec, dy = dxvec, dz = dxvec)
    mesh = (mesh + start)
    # xyz = mesh.cellCenters.value
    # x = order_mesh(xyz[0], mesh.nx)
    # y = order_mesh(xyz[1], mesh.ny)
    # z = order_mesh(xyz[2], mesh.nz)
    return mesh#, [x, y, z]




#%% 1.a  establish delta

def gen_uniform_delta(number_of_cells, length = 1.0):
    import math
    #digits = int(math.log10(number_of_cells))+1
    delta_x= length/number_of_cells #round(1/number_of_cells, digits)
    logger.info(f'Uniform: number of cells = {number_of_cells}, delta_x={delta_x} ')
    if delta_x < 4*10**-3:
        logger.debug(f'delta_space = {delta_x} < 4*10**-3')
    else:
        logger.critical(f'delta_space = {delta_x} > 4*10**-3')
    #assert(delta_x < 4*10**-3)
    return delta_x

def gen_nonuniform_central_segment_delta(number_of_cells, length = 1.0):
    assert(length > 0)
    L1, L2, L3 = length*0.25, length*0.50, length*0.25
    assert(L1+L2+L3 == length)
    n1, n3 = int(0.10*number_of_cells), int(0.10*number_of_cells)
    n2 = number_of_cells - n1 - n3
    assert(n2>0 and n1>0 and n3>0)
    dx1, dx2, dx3 = L1/n1, L2/n2, L3/n3
    delta_space=min(dx1, dx2, dx3)
    delta_space_max=max(dx1, dx2, dx3)
    logger.info(f'Segment non-uni: number of cells = {number_of_cells}, delta_min={delta_space},delta_max={delta_space_max}')
    if delta_space < 4*10**-3:
        logger.debug(f'delta_space = {delta_space} > 4*10**-3')
    else:
        logger.critical(f'delta_space = {delta_space} > 4*10**-3')
    #assert(delta_space < 4*10**-3)
    delta_vect = [dx1]*n1 + [dx2]*n2 + [dx3]*n3
    return delta_vect, delta_space


def gen_nonuniform_segment_delta(number_of_cells, length = 1.0):
    assert(length > 0)
    L1, L2, L3 = length*0.43, length*0.34, length*0.23
    assert(L1+L2+L3 == length)
    n1, n3 = int(0.07*number_of_cells), int(0.25*number_of_cells)
    n2 = number_of_cells - n1 - n3
    assert(n2>0 and n1>0 and n3>0)
    dx1, dx2, dx3 = L1/n1, L2/n2, L3/n3
    delta_space=min(dx1, dx2, dx3)
    delta_space_max=max(dx1, dx2, dx3)
    logger.info(f'Segment non-uni: number of cells = {number_of_cells}, delta_min={delta_space},delta_max={delta_space_max}')
    if delta_space < 4*10**-3:
        logger.debug(f'delta_space = {delta_space} > 4*10**-3')
    else:
        logger.critical(f'delta_space = {delta_space} > 4*10**-3')
    #assert(delta_space < 4*10**-3)
    delta_vect = [dx1]*n1 + [dx2]*n2 + [dx3]*n3
    return delta_vect, delta_space

def gen_nonuniform_chebyshev_delta(number_of_cells, length = 1.0 ):
    import numpy as np
    assert(length > 0)
    xmin=0.0 # it is infuneltal since there is a diff
    xmax=length
    # This function calculates the n Chebyshev points
    number_of_cells=number_of_cells+1 # one lost becaouse of diff
    ns = np.arange(1,number_of_cells+1)
    x = np.cos((2*ns-1)*np.pi/(2*number_of_cells))
    inverse_verc = (xmin+xmax)/2 + (xmax-xmin)*x/2 #(xmin+xmax)/2 dose note infulece dvec
    dxvec = inverse_verc[:-1]-inverse_verc[1:]
    delta_space=min(dxvec)
    delta_space_max=max(dxvec)
    logger.info(f'Chebyshev: number of cells = {number_of_cells-1}, delta_min={delta_space},delta_max={delta_space_max}')
    assert(delta_space>0.0)
    return dxvec, delta_space


def gen_nonuniform_inverse_chebyshev_delta(number_of_cells, length = 1.0 ):
    dxvec, delta_space = gen_nonuniform_chebyshev_delta(number_of_cells, length)
    #TODO invert dxvec
    #https://www.reddit.com/r/learnpython/comments/gma0is/splitting_an_array_into_two_and_swapping_the/
    
    assert (number_of_cells == len (dxvec))
    #n=number_of_cells//2
    #dxvec[:n],dxvec[n:] = dxvec[n:].copy(), dxvec[:n].copy()  # split_swap_vec
    return dxvec, delta_space





#%% Test single mesh extraction


if __name__ == '__main__':
    import matplotlib.pyplot as pt
    import numpy as np
    num_of_cells = 300
    delta_space = gen_uniform_delta(num_of_cells, length=2)    

    from time import time
    start = time()
    #code here
    mesh =gen_xyz_mesh(num_of_cells, delta_space, start=-1)
    print(f'Time taken to run: {time() - start} seconds')

    xyz = mesh.cellCenters.value
    start = time()
    x1 = reorder_mesh(xyz[0], mesh.nx)
    print(f'Time taken to run order_mesh: {time() - start} seconds')
    
    start = time()
    x2 =np.unique(xyz[0])
    print(f'Time taken to run unique: {time() - start} seconds')

    start = time()
    x3 = list(dict.fromkeys(xyz[0]))
    print(f'Time taken to run list(dict: {time() - start} seconds')

    #np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(x2, x3)



#%% Test single mesh extraction

if __name__ == '__main__':
    import matplotlib.pyplot as pt
    mesh = fi.Grid3D(nx = 3, ny = 2, nz = 1, dx = 0.5, dy = 2., dz = 4.)

    print(fi.numerix.nonzero(mesh.facesTop)[0])

    xyz = mesh.cellCenters.value
    x = reorder_mesh(xyz[0], mesh.nx)
    y = reorder_mesh(xyz[1], mesh.ny)
    z = reorder_mesh(xyz[2], mesh.nz)


    #np.unique(xyz[0])



#%% Test 1 

if __name__ == '__main__':
    number_of_cells = 10    
    delta_vect, delta_space  = gen_nonuniform_chebyshev_delta(number_of_cells, 2.0)
    print(delta_vect)
    mesh=gen_x_nnunif_mesh(delta_vect, start=0.0)
    #print (x)

    delta_vect, delta_space  =  gen_nonuniform_inverse_chebyshev_delta(number_of_cells, 2.0)
    print(delta_vect)
    mesh=gen_x_nnunif_mesh(delta_vect, start=0.0)
    #print (x)



#%% Test 1 
#TODO to debug!!!!
if __name__ == '__main__':
    number_of_cells = 50
    delta_vect, delta_space = gen_nonuniform_segment_delta(number_of_cells)
    mesh=gen_x_nnunif_mesh(delta_vect)
    x = array_from_mesh(1, mesh)
    print (x)

    delta_vect, delta_space = gen_nonuniform_segment_delta(number_of_cells, length=2)
    mesh=gen_x_nnunif_mesh(delta_vect, start=-1)
    x = array_from_mesh(1, mesh)
    print (x)

#%% Test 1 

if __name__ == '__main__':
    number_of_cells = 10
    delta_space = gen_uniform_delta(number_of_cells)
    mesh =gen_x_mesh(number_of_cells, delta_space)
    x = array_from_mesh(1, mesh)
    print (x)
    delta_space = gen_uniform_delta(number_of_cells, length=2)
    mesh =gen_x_mesh(number_of_cells, delta_space, start=-1)
    x = array_from_mesh(1, mesh)
    print (x)

    delta_space = gen_uniform_delta(number_of_cells, length=2)
    mesh =gen_x_mesh(number_of_cells, delta_space, start=-1)
    x = array_from_mesh(1, mesh)
    print (x)

    mesh  = gen_xy_mesh(number_of_cells, delta_space, start=-2)
    d_vector_mesh = array_from_mesh(2, mesh)
    print (d_vector_mesh)

    mesh = gen_xy_mesh(number_of_cells, delta_space, start=[[1], [4]])
    d_vector_mesh = array_from_mesh(2, mesh)
    print (d_vector_mesh)





# %%
