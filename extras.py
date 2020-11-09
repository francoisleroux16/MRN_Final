# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:37:28 2020

@author: Francois le Roux
This script is for all extra stuff
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

def cubic(x,val1,val2):
    tck = interpolate.splrep(val1,val2)
    return interpolate.splev(x,tck)

def cubic_spline(x,val1,val2):
    cs = interpolate.CubicSpline(val1, val2)
    xvals = cs(x)
    return xvals

def addit(x,y,**kwargs):
    if 'cpy' in kwargs:
        cpy = kwargs['cpy']
    else:
        cpy = 0
        
    if 'cpp' in kwargs:
        cpp = kwargs['cpp']
    else:
        cpp = 0
    
    return np.sin(cpy)**(cpp)

def initial_values():
    v = (36.5+34.4)/2 #For big windtunnel dataset = WindTunnel30
    P = ((7.9+7.0)/2)*100 #For big windtunnel dataset = WindTunnel30
    Pt = 87000 + P
    return v, P, Pt

def get_rho(p,v):
    '''p = delta p'''
    return 1/((v**2)/(2*p))

def get_v(p,rho):
    '''p = delta p'''
    return np.sqrt((2*p)/rho)


def spatial_resolution23(sorted_dictionary,data,d):
    '''
    Use this for spatial resolution along the horizontal axis

    Parameters
    ----------
    sorted_dictionary : Dictionary
        A dictionary sorted according to y-coordinates (i.e increasing from low to high)
    data : Numpy array
        Data collected by the probe, in the correct order.
    d : float
        Diameter of the head of the five-hole probe.

    Returns
    -------
    data : numpy array
        The same order and size as the input data, but with spatial resolution applied to it.

    '''
    index = np.lexsort((-data[:,0],data[:,1]))
    index_counter = 0
    for j in sorted_dictionary:
        counter = 0
        vals = np.zeros((len(sorted_dictionary[j]),3))
        for k in sorted_dictionary[j].keys():
            vals[counter,0] = abs(sorted_dictionary[j][k]['Z'])
            vals[counter,1] = sorted_dictionary[j][k]['P2']
            vals[counter,2] = sorted_dictionary[j][k]['P3']
            counter += 1
        P2new = cubic(vals[:,0]+d,vals[:,0],vals[:,1]) #remember order is according to y
        P3new = cubic(vals[:,0]-d,vals[:,0],vals[:,2])
        for j in range(len(P2new)):
            data[index[index_counter],3] = P2new[j]
            data[index[index_counter],4] = P3new[j]
            index_counter += 1
    return data

def spatial_resolution45(sorted_dictionary,data,d):
    '''
    Use this for spatial resolution along the vertical axis

    Parameters
    ----------
    sorted_dictionary : Dictionary
        A dictionary sorted according to Z-coordinates (i.e increasing from low to high) -> taking sorted dictionaries (over Z then Y) as input
    data : Numpy array
        Data collected by the probe, in the correct order.
    d : float
        Diameter of the head of the five-hole probe.

    Returns
    -------
    data : numpy array
        The same order and size as the input data, but with spatial resolution applied to it.

    '''
    index = np.lexsort((data[:,1],-data[:,0])) #sort z then y
    index_counter = 0
    for k in sorted_dictionary:
        counter = 0
        vals = np.zeros((len(sorted_dictionary[k]),3))
        for l in sorted_dictionary[k].keys():
            vals[counter,0] = sorted_dictionary[k][l]['Y']
            vals[counter,1] = sorted_dictionary[k][l]['P4']
            vals[counter,2] = sorted_dictionary[k][l]['P5']
            counter += 1
        P4new = cubic(vals[:,0]-d, vals[:,0], vals[:,1])
        P5new = cubic(vals[:,0]+d,vals[:,0],vals[:,2])
        for j in range(len(P4new)):
            data[index[index_counter],5] = P4new[j]
            data[index[index_counter],6] = P5new[j]
            index_counter += 1
    return data

# for j in range(5):
#     plt.figure('P'+str(j+1))
#     plt.title('Plot of variation of Pressure at probe hole {}'.format(str(j+1)))
#     plt.xlabel('Sample Number in Pass')
#     plt.ylabel('Magnitude of the Pressure [Pa]')
#     for k in range(30):
#         plt.plot(our_number12[30*k:30*k+30,j+2])