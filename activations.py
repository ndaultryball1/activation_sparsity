import numpy as np
from scipy.integrate import simpson, dblquad

# Activation functions

def soft( x, tau ):
    if( x < -tau ):
        return( x + tau )
    elif( np.less_equal( abs( x ), tau ) ):
        return( 0 )
    elif( x > tau ):
        return( x - tau )

def hard( x, tau ):
    if( np.less_equal( abs( x ), tau ) ):
        return( 0 )
    else:
        return( x )

# Length maps

def q( activation_func, prev, sigma_b, sigma_w, tau ):
    range = 100
    step = 0.1
    z = np.arange( -range, range, step )
    res = sigma_b ** 2 + sigma_w ** 2 * ( 1 / np.sqrt( 2 * np.pi ) ) * simpson( np.square( np.array( [ activation_func( np.sqrt( prev ) * point, tau ) for point in z ] ) ) * np.exp( -np.square( z ) / 2 ), z )    
    return( res )

def soft_q( prev, sigma_b, sigma_w, tau ):
    res = q( soft, prev, sigma_b, sigma_w, tau )
    return( res )

def hard_q( prev, sigma_b, sigma_w, tau ):
    res = q( hard, prev, sigma_b, sigma_w, tau )
    return( res )

# Correlation maps
def u1( q11, z1 ):
    return( np.sqrt( q11 ) * z1 )

def u2( prev, q22, z1, z2):
    res = np.sqrt( q22 ) * ( prev * z1 + np.sqrt( 1 - prev ** 2 ) * z2 )
    return( res )

def c( activation_func, prev, q11, q22, sigma_w, sigma_b ):
    def integrator( z1, z2 ):
        inner = activation_func( u1( q11, z1 ) ) * activation_func( u2( prev, q22, z1, z2 ) )
        integrand = inner * np.exp( - ( z1 ** 2 + z2 **2 ) / 2 )
        return( integrand )
    z1_range = np.inf
    z2_range = np.inf
    integral = dblquad( integrator, -z1_range, z1_range, -z2_range, z2_range )
    res = 1 / ( 2 * np.pi ) * sigma_w ** 2 * integral + sigma_b ** 2
    return( res )