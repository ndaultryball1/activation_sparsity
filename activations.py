import numpy as np
import scipy
from scipy.integrate import simpson, dblquad
from scipy.special import erfc

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
    # Length map by doing the integral
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

def V_soft_analytic( q, sigma_b, sigma_w, tau):
    # Analytic length map for soft thresholding
    x = tau / np.sqrt( q )
    # res = 1 / ( 2 * np.sqrt( 2 ) ) * q * np.square( sigma_w ) * ( erfc( x ) + ( 2 / np.sqrt( np.pi ) ) * x * np.exp ( - x ** 2) ) + sigma_b ** 2
    res = 2 * sigma_w ** 2 * ( ( q + tau ) ** 2 *( 1 - scipy.stats.norm.cdf( x ) ) - (( np.sqrt( q ) * tau ) / np.sqrt( 2 * np.pi ) )* np.exp( - x ** 2 / 2 ) ) + sigma_b ** 2
    return( res )

# Correlation maps
def u1( q11, z1 ):
    return( np.sqrt( q11 ) * z1 )

def u2( prev, q22, z1, z2):
    res = np.sqrt( q22 ) * ( prev * z1 + np.sqrt( 1 - prev ** 2 ) * z2 )
    return( res )

def correlation_map( activation_func, prev, q11, q22,  sigma_b, sigma_w, tau ):
    def integrator( z1, z2 ):
        inner = activation_func( u1( q11, z1 ), tau ) * activation_func( u2( prev, q22, z1, z2 ), tau )
        integrand = inner * np.exp( - ( z1 ** 2 + z2 **2 ) / 2 )
        return( integrand )
    z1_range = np.inf
    z2_range = np.inf
    integral = dblquad( integrator, -z1_range, z1_range, -z2_range, z2_range )[ 0 ]
    res = ( 1 / ( 2 * np.pi ) ) * sigma_w ** 2 * integral + sigma_b ** 2
    return( res )