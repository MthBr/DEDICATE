# -*- coding: utf-8 -*-
"""
Version 1, smoothing the Dirac delta function 
 as is done with diffusion interface methods 
 see equations 11, 12 and 13:
 https://doi.org/10.1016/j.jcp.2004.09.018
AND
Eq 13-17 in


@author: enzo
"""

#%% import pakages
import fipy as fi
import numpy as np


from dedicate_code.custom_funcs import  get_logger, benchmark
from dedicate_code.config import log_dir
global logger
if __name__ == '__main__':
    logger = get_logger(log_dir/'delta_functs.log', 'general')
else:
    logger = get_logger()




#%% Load functions

def delta_cos_func(mesh, x0, h, dim, m=2):
    #4-point cosine function
    return delta_func(mesh, x0, h, dim, approx_type= '2-cos' ,m=2)

def delta_l_func(mesh, x0, h, dim, m=1):
    #2-point hat function
    return delta_func(mesh, x0, h, dim, approx_type= '1-l*' ,m=1)



def delta_cubic_func(mesh, x0, h, dim, m=2):
    return delta_func(mesh, x0, h, dim, approx_type= 'Cubic' ,m=2)


def delta_LL_func(mesh, x0, h, dim, m=2):
    return delta_func(mesh, x0, h, dim, approx_type= 'LL' ,m=2)


#%% Define known deltas
def hat_funct(x, h, m=2):
    # m = 1 o 2
    # m =  moment is the support
    #2or4..-point hat function
    epsilon = m*h
    where_delta= (abs(x) <= epsilon)
    phi = 1./(m**2) * np.minimum(x/h + m, m - x/h)
    delta =  where_delta * 1./h * phi
    return delta


def cos_funct(x, h, m=2):
    fpnp = fi.numerix
    # m = 1 o 2
    # m =  moment is the support
    #2or4..-point cosine function
    epsilon = m*h
    where_delta= (abs(x) <= epsilon)
    ra = abs(x/h).value
    phi = 1./2 * (1+ fpnp.cos(fpnp.pi * ra))
    delta =  where_delta * 1./h * phi
    return delta


#%% Polinomial Legendre type delta, with support 1!!!!
# 
def faster_polyval(p, x):
    y = np.zeros(x.shape, dtype=float)
    for i, v in enumerate(p):
        y *= x
        y += v
    return y


def approx_legendre(approx_type, x, h, support = 1):
    fpnp = fi.numerix
    Pi=fpnp.pi

    epsilon = support*h
    where_delta= (abs(x) <= epsilon)

    ra = abs(x/h).value
    mask = where_delta #(ra <= 1)  #equivalent solution!
    phi = np.zeros_like(ra)
    r = ra[mask]

    if approx_type.startswith('l-1-0-1d'):
        phi[mask] =  1/2
    elif approx_type.startswith('l-1-1-1d'):
        phi[mask] =  1 - r  # equivalent hat!
    elif approx_type.startswith('l-2-2-1d'):
        coef = [15, -18, 9/2] #Add - 
        phi[mask] =  np.polyval(coef, r)
        #phi[mask] =  9/2-18*r+15*r**2
    elif approx_type.startswith('l-2-3-1d'):
        coef = [-30, 60, -36,  6] #TODO added - /6:molto basso   (-:alto)
        phi[mask] =  np.polyval(coef, r) #*(-1/3)

    elif approx_type.startswith('l-2-5-1d'):
        coef = [168, -945/2, 450, -150, 0, 9/2] #:- alto(0.5) /2: vicno(0.27)
        phi[mask] =   np.polyval(coef, r)  #*(-1/2)

    elif approx_type.startswith('l-1-1-2d'): #TODO add /3?
        phi[mask] =   (6/Pi)*(3-4*r) #*(1/4) #(/3 /4 cade in basso under.. standard:over)

    elif approx_type.startswith('l-1-2-2d'):#TODO add /3 ?
        coef = [5, -8, 3] #TODO (/3 si avvicina dall'alto) /4:un poco sotto /pi troppo sotto
        phi[mask] =   (12/Pi)*np.polyval(coef, r) 

    elif approx_type.startswith('l-2-2-2d'): #TODO errore!!?!?!
        coef = [15, -20, 6] #-/4: 
        phi[mask] =   12/Pi*np.polyval(coef, r) #* (-1/4)

    elif approx_type.startswith('l-2-3-2d'):
        coef = [7, -15, 10, -2] #-/10
        phi[mask] =   (-60/Pi)*np.polyval(coef, r)  #*(-1/10)

    elif approx_type.startswith('l-2-5-2d'):
        coef = [24, -70, 70, -25, 0, 1] #-/10
        phi[mask] =   (84/Pi)*np.polyval(coef, r)

    else:
        raise NameError(f'Not a valid approx_type:  {approx_type}')
       
    delta =  where_delta * 1./h * phi
    # h == delta_space
    if h*support < 9.9*10**-3:
        logger.debug(f'({support}=m)*epsilon= {h*support} < {9.9*10**-3}')
    else:
        logger.critical(f'({support}=m)*epsilon= {h*support} > {9.9*10**-3}')
    #assert(h*moment < 9.9*10**-3)

    return delta


#%%
def approx_trig(approx_type, x, h, support = 1):
    fpnp = fi.numerix
    Pi=fpnp.pi

    epsilon = support*h
    where_delta= (abs(x) <= epsilon)

    ra = abs(x/h).value
    mask = (ra <= 1)
    phi = np.zeros_like(ra)
    r = ra[mask]

    if approx_type.startswith('cos-1-1d'):
        phi[mask] =  1./2 * (1.+ fpnp.cos(Pi * r)) 
    elif approx_type.startswith('cos-2-1d'):
        phi[mask] =  1./2 * (1.+ fpnp.cos(Pi * r)) #TODO
    elif approx_type.startswith('cos-2-2d'):
        phi[mask] =  1./2 * (1.+ fpnp.cos(Pi * r)) #TODO
    else:
        raise NameError(f'Not a valid approx_type:  {approx_type}')
       
    delta =  where_delta * 1./h * phi
    # h == delta_space
    if h*support < 9.9*10**-3:
        logger.debug(f'({support}=m)*epsilon= {h*support} < {9.9*10**-3}')
    else:
        logger.critical(f'({support}=m)*epsilon= {h*support} > {9.9*10**-3}')
    #assert(h*moment < 9.9*10**-3)

    return delta



def approx_adf(approx_type, x, h, support = 2, z=0):
    epsilon = support*h
    ra = abs(x/epsilon).value
    mask = (ra <= 1)
    phi = np.zeros_like(ra)
    r = ra[mask]

    if approx_type.startswith('adf-0'):
        #N=1*1
        phi[mask] =  1./2

    elif approx_type.startswith('adf-1'):
        #N=2*1
        phi[mask] =  1./2 + 3/2*z*r

    elif approx_type.startswith('adf-2'):
        #N=2*2
        coef= [(45*z**2-15)/8, 3*z/2, (9-15*z**2)/8]
        phi[mask] =  np.polyval(coef, r) 

    elif approx_type.startswith('adf-3'):
        #N=3*2
        coef= [(175*z**3-105*z)/8, (45*z**2-15)/8, (75*z-105*z**3)/8, (9-15*z**2)/8 ]
        phi[mask] =  np.polyval(coef, r) 

    elif approx_type.startswith('adf-4'):
        #N=3*2
        coef= [(945-9450*z**2+11025*z**4)/128, \
                (175*z**3-105*z)/8,\
                -(525-4410*z**2 + 4725*z**4)/64,\
                -(105*z**3-75*z)/8,\
                (225-1050*z**2+945*z**4)/128
        ]
        phi[mask] =  np.polyval(coef, r) 

    else:
        raise NameError(f'Not a valid approx_type:  {approx_type}')


    N = 1
    where_delta= (abs(x) <= epsilon*N)

    delta =  where_delta * 1./(epsilon) * phi
    # h == delta_space

    if h*support < 9.9*10**-3:
        logger.debug(f'({support}=m)*epsilon= {h*support} < {9.9*10**-3}')
    else:
        logger.critical(f'({support}=m)*epsilon= {h*support} > {9.9*10**-3}')
    #assert(h*moment < 9.9*10**-3)

    return delta







#%% Piecewise Immersed Boundary
KK = (38-np.sqrt(69))/60
def phi5k(r):
    #err on paper -12600 instead of- 12440 
    #beta=np.polyval([-12600*KK**2, 0, 3600*KK**2], r) + np.polyval([-8400*KK, 0, 25680*KK, 0, -6840*KK+3123], r)
    #gamma = -40*r**2 * np.polyval([35, 0, -202, 0, 311], r)
    beta_gamma_coef=[- 1400, 0, 8080- 8400*KK, 0,- 12440+ 25680*KK- 12600*KK**2, 0, 3123 - 6840*KK + 3600*KK**2]
    beta_gamma=np.polyval(beta_gamma_coef, r)
    # 3123 - 6840*KK + 3600*KK**2 - 12440*r**2 + 25680*KK*r**2 - 12600*KK**2*r**2 + 8080*r**4 - 8400*KK*r**4 - 1400*r**6
    #(3123 - 6840*KK + 3600*KK.^2 - 12440*r.^2 + 25680*KK*r.^2 - 12600*KK.^2*r.^2 + 8080*r.^4 - 8400*KK*r.^4 - 1400*r.^6)
    return (136 - 40*KK - 40*r**2 + np.sqrt(2)*np.sqrt(beta_gamma))/280



def flex6pt(r,K):
    # as in 
    # a
    alpha=28
    beta=(9/4)-(3/2)*(K+r**2)+((22/3)-7*K)*r-(7/3)*r**3
    gamma=(1/4)*( ((161/36)-(59/6)*K+5*K**2)*(1/2)*r**2 + (-(109/24)+5*K)*(1/3)*r**4 + (5/18)*r**6 )
    discr=beta**2-4*alpha*gamma
    pm3=(-beta+np.sign((3/2)-K)*np.sqrt(discr))/(2*alpha)
    return pm3




def approx_piecewise(approx_type, x, h):
    fpnp = fi.numerix
    Pi=fpnp.pi
    Pi4 = fpnp.pi/4
    Pi2 = fpnp.pi/2

    rha = abs(x/h).value
    rh = (x/h).value
    phi1 = np.zeros_like(rha)

    if approx_type == 'p2h-s1':
        # equivalent to 2point hat!
        # 
        support = 1
        mask1 = (rha <= 1.0)
        r = rha[mask1]
        phi1[mask1] =  1-r
        phi = phi1
    elif approx_type == 'p3-s15':
        # 3-point function Yang  - stnd3ptYang
        # \phi_3^{IB} from BringleyThesis
        # Standard 3-point Kernel for IBM https://github.com/stochasticHydroTools/IBMethod/blob/master/IBKernels/stnd3pt.m
        support = 1.5
        mask1 = (rha <= 0.5)
        r = rha[mask1]
        phi1[mask1] =  1./3.*(1+fpnp.sqrt(-3*r**2+1))

        mask2 = ((0.5 < rha) &  (rha<= 1.5))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] = 1./6.*(5-3*r-fpnp.sqrt(-3*(1-r)**2+1))
        #phi2[~mask2] = 0.0
        phi = phi1+phi2
    elif approx_type == 'p4h-s2':
        # equivalent to 4-point hat!
        # 
        support = 2
        mask1 = (rha <= 2.0)
        r = rha[mask1]
        phi1[mask1] =  0.5-0.25*r
        phi = phi1
    elif approx_type == 'p*2-s15':  #'1-l*'
        # smoothed 2-point hat function
        # 
        support = 1.5
        mask1 = (rha <= 0.5)
        r = rha[mask1]
        phi1[mask1] =  0.75 - r**2

        mask2 = ((0.5 < rha) &  (rha<= 1.5))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] = np.polyval([0.5, - 1.5, 1.125], r)
        phi = phi1+phi2
    elif approx_type == 'p4-s2': #TODO err???
        # 4-point function Yang Lee  stnd4ptYang
        # the one suggested by Peskin
        # \phi_4^{IB} from BringleyThesis
        # Standard 4-point Kernel for IBM same as https://github.com/stochasticHydroTools/IBMethod/blob/master/IBKernels/stnd4pt.m
        support = 2
        mask1 = (rha <= 1.0)
        r = rha[mask1]
        phi1[mask1] =  1./8.*(3-2*r+fpnp.sqrt(np.polyval([-4, +4, +1], r)))

        mask2 = ((1.0 < rha) &  (rha<= 2.0))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] = 1./8.*(5-2*r-fpnp.sqrt(np.polyval([-4, +12, -7], r)))
        #phi2[~mask2] = 0.0
        phi = phi1+phi2
    elif approx_type == 'p-cubic-s2':  #'p2-Cubic'
        #piecewise cubic function #moment = 2  
        # phi_4^M BringleyThesis
        support = 2.0 
        mask1 = (rha <= 1.0)
        r = rha[mask1]
        phi1[mask1] =  np.polyval([0.5, -1, - 0.5, 1.], r)

        mask2 = ((1.0 < rha) &  (rha<= 2.0))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] = np.polyval([-1./6., +1, - 11./6., 1.], r) #?[-1./6., -1, - 11./6., 1.] on Tonberg
        phi = phi1+phi2

    elif approx_type == 'p-o5-s3':  
        # phi_6^M BringleyThesis
        support = 3.0 
        mask1 = (rha <= 1.0)
        r = rha[mask1]
        phi1[mask1] =  np.polyval([-1/12., 1/4, 5/12, -5/4, -1/3, 1], r)

        mask2 = ((1.0 < rha) &  (rha<= 2.0))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] = np.polyval([1/24, -3/8., 25/24, -5/8, -13/12, 1], r) #?[-1./6., -1, - 11./6., 1.] on Tonberg
        
        mask3 = ((2.0 < rha) &  (rha<= 3.0))
        phi3 = np.zeros_like(rha)
        r = rha[mask3]
        phi3[mask3] = np.polyval([-1/120, 1/8, -17/24, +15/8, -197/60, 1 ], r) #?[-1./6., -1, - 11./6., 1.] on Tonberg
        phi = phi1+phi2+phi3
        
    elif approx_type == 'p-LL-s2': 
        # on Tonberg
        #
        support = 2.0 
        mask1 = (rha <= 1.0)
        r = rha[mask1]
        phi1[mask1] =  1./12. * (14-15*r) #*1/2

        mask2 = ((1.0 < rha) &  (rha<= 2.0))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] = 1./12. * (2-r) #*1/2
        phi = phi1+phi2
    
    elif approx_type == 'p*3-s2':
        #smoothed 3-point function
        support =  2.0
        sq3 = fpnp.sqrt(3)
        sq3P108=sq3*Pi/108

        mask1 = (rha <= 1.0)
        r = rha[mask1]
        phi1[mask1] = 17/48 + sq3P108 + r/4 - r**2/4 + (1-2*r)/16*(fpnp.sqrt(-12*r**2+12*r+1)) - sq3/12*fpnp.arcsin(sq3/2*(2*r-1))
       
        mask2 = ((1.0 < rha) &  (rha<= 2.0))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] =55/48 - sq3P108 -13*r/12 +r**2/4 + (2*r-3)/48*(fpnp.sqrt(-12*r**2+36*r-23)) + sq3/36*fpnp.arcsin(sq3/2*(2*r-3))
        phi = phi1+phi2

    elif approx_type == 'p*4-s25':
        #smoothed 4-point piecewise function
        #
        support =  2.5
        sq2 = fpnp.sqrt(2)
        mask1 = (rha <= 0.5)
        r = rha[mask1]
        phi1[mask1] = 3/8 + Pi/32 - r**2/4
        mask2 = ((0.5 < rha) &  (rha<= 1.5))
        phi2 = np.zeros_like(rha)
        r = rha[mask2]
        phi2[mask2] =1/4+ (1-r)/8*(np.sqrt(-2 + 8*r -4*r**2)) -1/8*np.arcsin(sq2*(r-1))
        mask3 = ((1.5 < rha) &  (rha<= 2.5))
        phi3 = np.zeros_like(rha)
        r = rha[mask3]
        phi3[mask3] = 17/16-Pi/64-3*r/4+r**2/8+(r-2)/16*(np.sqrt(-14+16*r-4*r**2)) + 1/16*np.arcsin(sq2*(r-2))
        phi = phi1+phi2+phi3

    elif approx_type == 'pg5-s25':
        #5-point piecewise Gaussian-Like function
        #
        support =  2.5
        mask1 = (rha <= 0.5)
        r = rha[mask1]
        phi1[mask1] = phi5k(r)


        mask20 = ((-1.5 < rh) &  (rh<= -0.5))
        phi20 = np.zeros_like(rh)
        r = rh[mask20]
        r=r+1
        phi20[mask20] = (-4*phi5k(r) - 3*KK*r- KK+ np.polyval([-1,-1,4,4], r))/6

        mask21 = ((0.5 < rh) &  (rh<= 1.5))
        phi21 = np.zeros_like(rh)
        r = rh[mask21]
        r=r-1
        phi21[mask21] = (-4*phi5k(r) + 3*KK*r-KK+ np.polyval([1,-1,-4,4], r))/6
        

        mask30 = ((-2.5 < rh) &  (rh<= -1.5))
        phi30 = np.zeros_like(rh)
        r = rh[mask30]
        r=r+2
        phi30[mask30] = (2*phi5k(r) + 3*KK*r+ 2*KK+ np.polyval([1,2,-1,-2], r))/12

        mask31 = ((1.5 < rh) &  (rh<= 2.5))
        phi31 = np.zeros_like(rh)
        r = rh[mask31]
        r=r-2
        phi31[mask31] = (2*phi5k(r) - 3*KK*r+ 2*KK+ np.polyval([-1,2, 1,-2], r))/12

        phi = phi1+phi20+phi21+phi31+phi30

    elif approx_type == 'pg6-s3':
        K = 59/60-np.sqrt(29)/20
        #OR K= 0
        support =  3.0

        mask1 = ((0.0 < rh) &  (rh<= 1.0))
        r = rha[mask1]
        pm3 = flex6pt(r,K)
        phi1[mask1] = 2*pm3 + (5/8)  - (1/4)*(K+r**2)

        mask01 = ((-1.0< rh) &  (rh<= 0.0))
        phi01 = np.zeros_like(rh)
        r = rh[mask01]
        r=r+1
        pm3 = flex6pt(r,K)
        phi01[mask01] = 2*pm3 + (1/4) +  (1/6)*(4-3*K)*r -  (1/6)*r**3

        mask02 = ((-2.0< rh) &  (rh<= -1.0))
        phi02 = np.zeros_like(rh)
        r = rh[mask02]
        r=r+2
        pm3 = flex6pt(r,K)
        phi02[mask02] = -3*pm3 - (1/16) + (1/8)*(K+r**2) + (1/12)*(3*K-1)*r + (1/12)*r**3


        mask03 = ((-3.0< rh) &  (rh<= -2.0))
        phi03 = np.zeros_like(rh)
        r = rh[mask03]
        r=r+3
        pm3 = flex6pt(r,K)
        phi03[mask03] = pm3

        mask12 = ((1.0< rh) &  (rh<= 2.0))
        phi12 = np.zeros_like(rh)
        r = rh[mask12]
        r=r-1
        pm3 = flex6pt(r,K)
        phi12[mask12] = -3*pm3 + (1/4)   -  (1/6)*(4-3*K)*r +  (1/6)*r**3

        mask13 = ((2.0< rh) &  (rh<= 3.0))
        phi13 = np.zeros_like(rh)
        r = rh[mask13]
        r=r-1
        pm3 = flex6pt(r,K)
        phi13[mask13] = pm3 - (1/16) + (1/8)*(K+r**2) - (1/12)*(3*K-1)*r - (1/12)*r**3

        phi = phi1 + phi01 + phi02 + phi03 + phi12+ phi13

    elif approx_type == 'p-cos-s2':
        #4-point cosine function
        support =  2.0
        mask = (rha <= 2.0)
        r = rha[mask]
        phi1[mask] = 1./4 * (1+ fpnp.cos(Pi2 * r))
        phi = phi1
    elif approx_type == 'p*cos-s25':  #'1-l*'
        # smoothed 4-point cosine function
        # 
        support = 2.5
        mask1 = (rha <= 1.5)
        phi1 = np.zeros_like(rha)
        r_m = rh[mask1]
        phi1[mask1] =  1/4 + 1/(2*Pi)* fpnp.sin(Pi2*r_m+Pi4) - 1/(2*Pi)*fpnp.sin(Pi2*r_m-Pi4)
        #phi1[~mask1] = 0.0

        mask2 = ((1.5 < rha) &  (rha<= 2.5))
        phi2 = np.zeros_like(rha)
        ra_m = rha[mask2]
        phi2[mask2] = 5/8 - ra_m/4 - 1/(2*Pi) *fpnp.sin(Pi2*ra_m-Pi4)
        #phi2[~mask2] = 0.0
        phi = phi1+phi2

    else:
        raise NameError(f'Not a valid approx_type:  {approx_type}')
    

    epsilon = support*h
    where_delta= (abs(x) <= epsilon)
    delta =  where_delta * 1./h * phi
    # h == delta_space
    if h*support < 9.9*10**-3:
        logger.debug(f'({support}=m)*epsilon= {h*support} < {9.9*10**-3}')
    else:
        logger.critical(f'({support}=m)*epsilon= {h*support} > {9.9*10**-3}')
    #assert(h*moment < 9.9*10**-3)

    return delta
















#%%

def approx_old(approx_type, x, h, m):
    fpnp = fi.numerix

    if approx_type == '1-l':
        #2-point hat function
        moment = m = 1
        epsilon = m*h
        where_delta= (abs(x) <= epsilon)
        phi = 1./(m**2) * np.minimum(x/h + m, m - x/h)
    elif approx_type == '2-l':
        #4-point hat function
        moment = m = 2
        epsilon = m*h
        where_delta= (abs(x) <= epsilon)
        phi = 1./(m**2) * np.minimum(x/h + m, m - x/h)
    elif approx_type == '1*-l':
        #smoothed 2-point hat function
        moment = m = 1
        epsilon = (m+0.5)*h
        xi = abs(x/h).value
        where_delta= (abs(xi) <= epsilon/h) #TODO does not work!!!
        phi =np.where(xi <= 0.5, \
            0.75 - xi**2, \
            1.125 - 1.5*xi + 0.5* xi**2)
    elif approx_type == '1-l*':
        #smoothed 2-point hat function
        moment = m = 1
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        xi = abs(x/h).value
        phi =np.where(xi <= 0.5, \
            0.75 - xi**2, \
            1.125 - 1.5*xi + 0.5* xi**2)
    elif approx_type == '1-cos':
        #4-point cosine function
        moment = m = 1
        epsilon = m*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        phi = 1./2 * (1+ fpnp.cos(fpnp.pi * ra))
    elif approx_type == '2-cos':
        #4-point cosine function
        moment = m = 2
        epsilon = m*h
        where_delta= (abs(x) <= epsilon)
        phi = 1./4 * (1+ fpnp.cos(fpnp.pi * x / (2*h)))
    elif approx_type == 'cos_altern':
        moment = m = 2
        epsilon = m*h
        where_delta= (abs(x) < epsilon)
        phi = (1. + fpnp.cos(fpnp.pi * x / epsilon)) / 2 / epsilon
    elif approx_type == '2*-cos':
        #smoothed 4-point cosine function - Class C^2
        moment = m = 2
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        r = (x/h).value
        Pi=fpnp.pi
        Pi4 = fpnp.pi/4
        Pi2 = fpnp.pi/2
        phi1 =np.where(ra <= 1.5, \
            1/4 + 1/(2*Pi)* fpnp.sin(Pi2*r+Pi4) - 1/(2*Pi)*fpnp.sin(Pi2*r-Pi4)  , \
                0.0 )
        phi2 =np.where((1.5 < ra) &  (ra<= 2.5), \
            5/8 - ra/4 - 1/(2*Pi) *fpnp.sin(Pi2*ra-Pi4) , \
             0.0)
        phi = phi1+phi2
    elif approx_type == '2*-cosTEST3':
        #smoothed 4-point cosine function - Class C^2
        moment = m = 2
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        r = (x/h).value
        Pi=fpnp.pi
        Pi4 = fpnp.pi/4
        Pi2 = fpnp.pi/2

        mask1 = (ra <= 1.5)
        phi1 = np.zeros_like(ra)
        r_m = r[mask1]
        phi1[mask1] =  1/4 + 1/(2*Pi)* fpnp.sin(Pi2*r_m+Pi4) - 1/(2*Pi)*fpnp.sin(Pi2*r_m-Pi4)
        #phi1[~mask1] = 0.0

        mask2 = ((1.5 < ra) &  (ra<= 2.5))
        phi2 = np.zeros_like(ra)
        ra_m = ra[mask2]
        phi2[mask2] = 5/8 - ra_m/4 - 1/(2*Pi) *fpnp.sin(Pi2*ra_m-Pi4)
        #phi2[~mask2] = 0.0
        phi = phi1+phi2
    elif approx_type == '2*-cosTEST':
        #smoothed 4-point cosine function - Class C^2
        moment = m = 2
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        r = (x/h).value
        Pi=fpnp.pi
        phi =np.where(ra <= 1.5, \
            1/(4*Pi)*( Pi + 2* fpnp.sin(Pi/4*(2*r+1)) - 2*fpnp.sin(Pi/4*(2*r-1)) )   , \
            -1/(8*Pi)*(-5*Pi+2*Pi*ra  + 4*fpnp.sin(Pi/4*(2*ra-1))) )
    elif approx_type == '2*-cosTEST2':
        #smoothed 4-point cosine function - Class C^2
        moment = m = 2
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        r = (x/h).value
        Pi=fpnp.pi
        Pi4 = fpnp.pi/4
        phi1 =np.where(ra <= 1.5, \
            1/(4*Pi)*( Pi + 2* fpnp.sin(Pi4*(2*r+1)) - 2*fpnp.sin(Pi4*(2*r-1)) ) , \
                0.0 )
        phi2 =np.where((1.5 < ra) &  (ra<= 2.5), \
            -1/(8*Pi)*(-5*Pi+2*Pi*ra  + 4*fpnp.sin(Pi4*(2*ra-1))) , \
             0.0)
        phi = phi1+phi2
    elif approx_type == '3*f-arc':
        #smoothed 3-point function - Class C^2
        moment = m = 3
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        sq3 = fpnp.sqrt(3)
        Pi=fpnp.pi
        sq3P108=sq3*Pi/108
        phi1 =np.where(ra <= 1.0, \
            17/48 + sq3P108 + ra/4 - ra**2/4 + (1-2*ra)/16*(fpnp.sqrt(-12*ra**2+12*ra+1)) - sq3/12*fpnp.arcsin(sq3/2*(2*ra-1)) , \
                0.0 )
        phi2 =np.where((1.0 < ra) &  (ra<= 2), \
            55/48 - sq3P108 -13*ra/12 +ra**2/4 + (2*ra-3)/48*(fpnp.sqrt(-12*ra**2+36*ra-23)) + sq3/36*fpnp.arcsin(sq3/2*(2*ra-3))      , \
             0.0)
        phi = phi1+phi2
    elif approx_type == '4*f':
        #smoothed 4-point piecewise function - Class C^2
        moment = m = 4
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)
        ra = abs(x/h).value
        sq2 = fpnp.sqrt(2)
        Pi=fpnp.pi
        phi1 =np.where(ra <= 0.5, \
            3/8 + Pi/32 - ra**2/4 , 0.0 )
        phi2 =np.where((0.5 < ra) &  (ra<= 1.5), \
            1/4+ (1-ra)/8*(fpnp.sqrt(-2 + 8*ra -4**2*ra)) -1/8*fpnp.arcsin(sq2*(ra-1)), \
             0.0)
        phi3 =np.where((1.5 < ra) &  (ra<= 2.5), \
            17/16-Pi/64-3*ra/4+ra**2/8+(ra-2)/16*(fpnp.sqrt(-14+16*ra-4**2*ra)) + 1/16*fpnp.arcsin(sq2*(ra-2)), \
             0.0)        
        phi = phi1+phi2+phi3
    elif approx_type == '4f':
        #smoothed 4-point piecewise function - Class C^2
        moment = m = 2
        epsilon = (m+0.5)*h
        where_delta= (abs(x) <= epsilon)

        ra = abs(x/h).value
        sq2 = fpnp.sqrt(2)
        Pi=fpnp.pi

        mask1 = (ra <= 0.5)
        phi1 = np.zeros_like(ra)
        ra_m = ra[mask1]
        phi1[mask1] = 3/8 + Pi/32 - ra_m**2/4
        phi1[~mask1] = 0.0

        mask2 = ((0.5 < ra) &  (ra<= 1.5))
        phi2 = np.zeros_like(ra)
        ra_m = ra[mask2]
        phi2[mask2] = 1/4+ (1-ra_m)/8*(np.sqrt(-2 + 8*ra_m -4*ra_m**2)) -1/8*np.arcsin(sq2*(ra_m-1))
        phi2[~mask2] = 0.0

        mask3 = ((1.5 < ra) &  (ra<= 2.5))
        phi3 = np.zeros_like(ra)
        ra_m = ra[mask3]
        phi3[mask3] = 17/16-Pi/64-3*ra_m/4+ra_m**2/8+(ra_m-2)/16*(np.sqrt(-14+16*ra_m-4*ra_m**2)) + 1/16*np.arcsin(sq2*(ra_m-2))
        phi3[~mask3] = 0.0 
        phi = phi1+phi2+phi3
    elif approx_type == '2-Cubic':
        moment = m = 2
        epsilon = m*h
        where_delta= (abs(x) <= epsilon)
        xi = abs(x/h).value
        phi =np.where(xi <= 1, \
            1. - 0.5*xi - xi**2 + 0.5 * xi**3 , \
            1. - 11./6.*xi + xi**2 - 1./6.* xi**3)
    elif approx_type == '2-LL':
        moment = m = 2
        epsilon = m*h
        where_delta = (abs(x) <= epsilon)
        xi = abs(x/h).value
        phi =np.where(xi <= 1, \
            1./12. * (14-15*xi), \
            1./12. * (2-xi) )
    else:
        raise NameError(f'Not a valid approx_type:  {approx_type}')
    
      
    delta =  where_delta * 1./h * phi
    # h == delta_space
    if h*moment < 9.9*10**-3:
        logger.debug(f'({moment}=m)*epsilon= {h*moment} < {9.9*10**-3}')
    else:
        logger.critical(f'({moment}=m)*epsilon= {h*moment} > {9.9*10**-3}')
    #assert(h*moment < 9.9*10**-3)

    return delta




#%%

def approx(approx_type, x, h, support=1):
    if approx_type.startswith('p'):
        delta = approx_piecewise(approx_type, x, h) #cannot change support, only epsilon(eqv. == h*support)
    
    elif approx_type.startswith('adf'):
        if approx_type.endswith('z0'):
            delta = approx_adf(approx_type, x, h, support=1, z=0.0)
        elif approx_type.endswith('z1'):
            delta = approx_adf(approx_type, x, h, support=1, z=1.0)
        else:
            delta = approx_adf(approx_type, x, h, support=1, z=1/4)
    elif approx_type.startswith('cos'):
        if approx_type.endswith('s2'):
            delta = approx_trig(approx_type, x, h, support=2)
        else:
            delta = approx_trig(approx_type, x, h, support)
    
    elif approx_type.startswith('l'):
        if approx_type.endswith('s2'):
            delta = approx_legendre(approx_type, x, h, support=2)
        else:
            delta = approx_legendre(approx_type, x, h, support)
    else:
        raise NameError(f'Not a valid approx_type:{approx_type}')
    return delta



def delta_radial_func(mesh, x0, h, dim, approx_type, support=2):
    #Initial Impulse condition
    # delta_space = h
    if dim == 1 : 
        xyz= abs(mesh.cellCenters[0]-x0)
    if dim == 2 : 
        xyz = (mesh.cellCenters[0]-x0)**2 + (mesh.cellCenters[1]-x0)**2
    if dim == 3 : 
        xyz = np.sqrt((mesh.cellCenters[0]-x0)**2 + (mesh.cellCenters[1]-x0)**2 + (mesh.cellCenters[2]-x0)**2)

    delt_values =  approx(approx_type, xyz, h, support)

    logger.debug(f'Delta {approx_type} Dirac non negative: {np.sum(np.array(delt_values.value) > 0)}')
    return delt_values.value


def delta_tensor_func(mesh, x0, epsilon, dim, approx_type, support=1):
    import numpy as np
    if dim == 1 :
        x = mesh.cellCenters[0]-x0  # mesh.x- x0
        delt_values = approx(approx_type, x, epsilon, support)
    logger.debug(f'did 1D layer')
    if dim == 2 : 
        x = mesh.cellCenters[0]-x0  # mesh.x- x0
        delt_values = approx(approx_type, x, epsilon, support)
        logger.debug(f'2D layer')
        y = mesh.y-x0  #mesh.cellCenters[1]-x0
        delt_values = delt_values * approx(approx_type, y, epsilon, support)
    if dim >= 3 : 
        x = mesh.cellCenters[0]-x0  # mesh.x- x0
        delt_values = approx(approx_type, x, epsilon, support)
        logger.debug(f'2D layer')
        y = mesh.y-x0  #mesh.cellCenters[1]-x0
        delt_values = delt_values * approx(approx_type, y, epsilon, support)
        logger.debug(f'add 3D layer')
        z = mesh.z-x0  #mesh.cellCenters[2]-x0
        delt_values = delt_values * approx(approx_type, z, epsilon, support)
    logger.debug(f'Delta {approx_type} Dirac non negative: {np.sum(np.array(delt_values.value) > 0)}')
    return delt_values.value

def delta_scale_tensor_func(mesh, x0, h, dim, approx_type, support=1):
    import numpy as np
    if dim == 1 :
        x = mesh.cellCenters[0]-x0  # mesh.x- x0
        delt_values = approx(approx_type, x, h, support)
    logger.debug(f'did 1D layer')
    if dim == 2 : 
        scale = np.sqrt(2)
        x = mesh.cellCenters[0]-x0  # mesh.x- x0
        delt_values = approx(approx_type, x, h/scale, support)
        logger.debug(f'2D layer')
        y = mesh.y-x0  #mesh.cellCenters[1]-x0
        delt_values = delt_values * approx(approx_type, y, h/scale, support)
    if dim >= 3 : 
        x = mesh.cellCenters[0]-x0  # mesh.x- x0
        delt_values = approx(approx_type, x, h, support)
        logger.debug(f'2D layer')
        y = mesh.y-x0  #mesh.cellCenters[1]-x0
        delt_values = delt_values * approx(approx_type, y, h, support)
        logger.debug(f'add 3D layer')
        z = mesh.z-x0  #mesh.cellCenters[2]-x0
        delt_values = delt_values * approx(approx_type, z, h, support)
    logger.debug(f'Delta {approx_type} Dirac non negative: {np.sum(np.array(delt_values.value) > 0)}')
    return delt_values.value


def delta_func(mesh, x0, epsilon, dim, approx_type, delta_product='tensor', support=1):
    if delta_product=='tensor':
        delt_values = delta_tensor_func(mesh, x0, epsilon, dim, approx_type, support)
    elif delta_product=='radial':
        delt_values = delta_radial_func(mesh, x0, epsilon, dim, approx_type, support)
    elif delta_product=='scaled_tensor':
        delt_values = delta_scale_tensor_func(mesh, x0, epsilon, dim, approx_type, support)  
    else:
        raise NameError(f'Not a valid delta_product:{delta_product}')
    return delt_values




@benchmark
def test_delta_func(mesh, x0, h, dim, approx_type, m=2):
    for _ in range(3):
        t = delta_func(mesh, x0, h, dim, approx_type, m)
    return t




#%% Test K phi5

if __name__ == '__main__':


    print((9-4*KK)/16)
    print(phi5k(0.5))
    print(phi5k(-0.5))

    print('Exterior')
    print((4*KK-1)/16)
    print(phi5k(3/2))




#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import array_from_mesh, gen_xyz_nnunif_mesh, gen_nonuniform_segment_delta
    dim = 3
    nxy = number_of_cells = 91
    delta_vect, delta_space = gen_nonuniform_segment_delta(number_of_cells, length=2)
    mesh =gen_xyz_nnunif_mesh(delta_vect, start=-1)
    xyz=array_from_mesh(3, mesh, len(delta_vect))
    epsilon = delta_space *1.4
    TEST =delta_func(mesh, 0.0, epsilon, dim, '2*-cosTEST') #TODO Just for testing Purpose
    values = delta_func(mesh, 0.0, epsilon, dim, '2*-cosTEST2')
    np.testing.assert_array_equal(TEST, values)
    TEST_3d=TEST.reshape((nxy, nxy, nxy))
    values_3d = values.reshape((nxy, nxy, nxy))

    ind = np.unravel_index(np.argmax(values_3d, axis=None), values_3d.shape)
    print(ind)
    print([nxy//2,nxy//2,nxy//2])

    from dedicate_code.data_exp.explore_utils_sols import compare_show_3d
    fig, axs = compare_show_3d(values_3d, TEST_3d, xyz, loc = ind, uniform= False)
    fig.show()






#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import array_from_mesh, gen_xyz_mesh, gen_uniform_delta
    dim = 3
    nxy = number_of_cells = 125
    delta_space = gen_uniform_delta(number_of_cells, length=2)
    mesh =gen_xyz_mesh(number_of_cells, delta_space, start=-1)
    xyz=array_from_mesh(3, mesh)
    epsilon = delta_space *1.4
    TEST =delta_func(mesh, 0.0, epsilon, dim, '2*-cos') #TODO Just for testing Purpose
    values = delta_func(mesh, 0.0, epsilon, dim, '2*-cosTEST2')
    np.testing.assert_array_equal(TEST, values)
    TEST_3d=TEST.reshape((nxy, nxy, nxy))
    values_3d = values.reshape((nxy, nxy, nxy))

    from dedicate_code.data_exp.explore_utils_sols import compare_show_3d
    fig, axs = compare_show_3d(values_3d, TEST_3d, xyz, loc = [nxy//2,nxy//2,nxy//2])
    fig.show()



#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_xyz_mesh, gen_uniform_delta
    dim = 3
    nxy = number_of_cells = 150
    delta_space = gen_uniform_delta(number_of_cells, length=2)
    mesh =gen_xyz_mesh(number_of_cells, delta_space, start=-1)
    epsilon = delta_space *1.4

    test_delta_func(mesh, 0.0, epsilon, dim, '2*-cos') #TODO Just for testing Purpose
    
    test_delta_func(mesh, 0.0, epsilon, dim, '2*-cosTEST2')





#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import array_from_mesh, gen_xyz_mesh, gen_uniform_delta
    dim = 3
    nxy = number_of_cells = 125
    delta_space = gen_uniform_delta(number_of_cells, length=2)
    mesh =gen_xyz_mesh(number_of_cells, delta_space, start=-1)
    xyz=array_from_mesh(3, mesh)
    epsilon = delta_space *1.4
    TEST =delta_func(mesh, 0.0, epsilon, dim, '3*f-arc') #TODO Just for testing Purpose
    values = delta_func(mesh, 0.0, epsilon, dim, '2-Cubic') # 2*-cos 2-Cubic
    #np.testing.assert_array_equal(TEST, values)
    TEST_3d=TEST.reshape((nxy, nxy, nxy))
    values_3d = values.reshape((nxy, nxy, nxy))

    from dedicate_code.data_exp.explore_utils_sols import compare_show_3d
    fig, axs = compare_show_3d(values_3d, TEST_3d, xyz, loc = [nxy//2,nxy//2,nxy//2])
    fig.show()






#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_x_mesh, gen_uniform_delta
    dim = 1
    number_of_cells = 300
    delta_space = gen_uniform_delta(number_of_cells, 2)
    mesh, x =gen_x_mesh(number_of_cells, delta_space, -1)
    epsilon = delta_space *1.4
    TEST =delta_func(mesh, 0.0, epsilon, dim, '2*-cosTEST') #TODO Just for testing Purpose
    values = delta_func(mesh, 0.0, epsilon, dim, '2*-cosTEST2')
    np.testing.assert_array_equal(TEST, values)

    import matplotlib.pyplot as plt
    plt.plot(x[0], TEST, label="1")
    plt.plot(x[0], values, linestyle="--", label="2")
    plt.show()



#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_x_mesh, gen_uniform_delta
    dim = 1
    number_of_cells = 300
    delta_space = gen_uniform_delta(number_of_cells, 2)
    mesh, x =gen_x_mesh(number_of_cells, delta_space, -1)
    epsilon = delta_space *1.4
    TEST =delta_func(mesh, 0.0, epsilon, dim, '1-l*') #TODO Just for testing Purpose
    values = delta_func(mesh, 0.0, epsilon, dim, '1*-l')
    np.testing.assert_array_equal(TEST, values)
    import matplotlib.pyplot as plt
    plt.plot(x[0], TEST)
    plt.plot(x[0], values)
    plt.show()


#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_x_mesh, gen_uniform_delta
    dim = 1
    number_of_cells = 150
    delta_space = gen_uniform_delta(number_of_cells)
    mesh, _ =gen_x_mesh(number_of_cells, delta_space)
    mesh = (mesh - [[0.5]])*4
    x = [mesh.x.value]
    epsilon = delta_space *number_of_cells

    phi = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
    #phi_t = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
    #phi_t[:] =delta_c_func_my(mesh, 0, epsilon*2, dim) #TODO Just for testing Purpose
    #phi[:] = delta_LL_func(mesh, 0, epsilon, dim) # delta_cos_func delta_l_func
    phi[:] = delta_cos_func(mesh, 0, epsilon, dim) # delta_cubic_func delta_LL_func

    #np.testing.assert_array_equal(phi_t.value, phi.value)
    import matplotlib.pyplot as plt
    #plt.plot(x[0], phi_t.value)
    plt.plot(x[0], phi.value)
    plt.show()
    print(min(phi.value), max(phi.value))


#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_x_mesh, gen_uniform_delta
    dim = 1
    number_of_cells = 400
    delta_space = gen_uniform_delta(number_of_cells)
    mesh =gen_x_mesh(number_of_cells, delta_space)
    x = array_from_mesh(1, mesh)
    epsilon = delta_space *5

    phi = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
    #phi_t = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
    #phi_t[:] =delta_c_func_my(mesh, 0, epsilon*2, dim) #TODO Just for testing Purpose
    phi[:] = delta_cos_func(mesh, 0.5, epsilon, dim) # delta_cos_func delta_l_func
    #phi[:] = delta_cubic_func(mesh, 0.5, epsilon, dim) # delta_cubic_func delta_LL_func

    #np.testing.assert_array_equal(phi_t.value, phi.value)
    import matplotlib.pyplot as plt
    #plt.plot(x[0], phi_t.value)
    plt.plot(x[0], phi.value)
    plt.show()
    print(min(phi.value), max(phi.value))

#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_x_mesh, gen_uniform_delta
    dim = 1
    number_of_cells = 125
    delta_space = gen_uniform_delta(number_of_cells)
    mesh =gen_x_mesh(number_of_cells, delta_space)
    x = array_from_mesh(1, mesh)
    epsilon = delta_space *10.
    TEST =delta_c_func_my(mesh, 0.5, epsilon, dim) #TODO Just for testing Purpose
    values = delta_c_func(mesh, 0.5, epsilon, dim)
    np.testing.assert_array_equal(TEST, values)
    import matplotlib.pyplot as plt
    plt.plot(x[0], TEST)
    plt.plot(x[0], values)
    plt.show()




#%%
if __name__ == '__main__':
    dim = 1
    number_of_cells = 5
    delta_space = 1
    mesh =gen_x_mesh(number_of_cells, delta_space)
    mesh = mesh - [[2.5]]
    x = [mesh.x.value]
    epsilon = delta_space *3.

    phi = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
    phi_t = fi.CellVariable(name="phi", mesh=mesh, hasOld=True,  value=0.)
    phi_t[:] =delta_c_func_my(mesh, 0, epsilon, dim) #TODO Just for testing Purpose
    phi[:] = delta_c_func(mesh, 0, epsilon, dim)

    np.testing.assert_array_equal(phi_t.value, phi.value)
    import matplotlib.pyplot as plt
    plt.plot(x[0], phi_t.value)
    plt.plot(x[0], phi.value)
    plt.show()








#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_xy_mesh, gen_uniform_delta
    dim = 2
    number_of_cells = 125
    delta_space = gen_uniform_delta(number_of_cells)
    mesh =gen_xy_mesh(number_of_cells, delta_space)
    epsilon = delta_space *2.
    TEST =delta_c_func_my(mesh, 0.5, epsilon, dim) #TODO Just for testing Purpose
    values = delta_c_func(mesh, 0.5, epsilon, dim)
    np.testing.assert_array_equal(TEST, values)
    import matplotlib.pyplot as plt
    plt.plot(x[0], TEST)
    plt.plot(x[0], values)
    plt.show()

#%%






if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_xyz_mesh, gen_uniform_delta
    dim = 3
    nxy = number_of_cells = 125
    delta_space = gen_uniform_delta(number_of_cells, length=2)
    mesh =gen_xyz_mesh(number_of_cells, delta_space, start=-1)
    
    epsilon = delta_space *1.4
    #TEST =delta_c_func_my(mesh, 0.5, epsilon, dim) #TODO Just for testing Purpose
    values = delta_cos_func(mesh, 0.5, epsilon, dim)
    #np.testing.assert_array_equal(TEST, values)

    
    import matplotlib.pyplot as plt
    plt.plot(xyz[0], values[nxy//2, nxy//2, :], label="1")
    #plt.plot(xyz[0], TEST[nxy//2, nxy//2, :], linestyle="--", label="h")
    plt.set_xlabel("z ; xy(0.5)")
    plt.set_ylabel("h")
    plt.show()






#%%
if __name__ == '__main__':
    from dedicate_code.data_etl.etl_utils import gen_x_nnunif_mesh, gen_nonuniform_segment_delta
    delta_vect, delta_space = gen_nonuniform_segment_delta(125)
    mesh =gen_x_nnunif_mesh(delta_vect)
    x = array_from_mesh(1, mesh)


    import matplotlib.pyplot as plt
    plt.plot(x[0], three_time_phi[1][-1])
    plt.show()


