import numpy as np
from math import erf, pi

def activation(x):
    return 0.5*x*(1 + erf(x/(2**0.5)))

def derivative(x):
    phi = 0.5*(1 + erf(x/(2**(0.5))))
    pdf = (np.exp(-(x**2)/2))/((2*pi)**0.5)
    return phi  + 0.5*x*pdf