# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:36:26 2020

@author: Francois le Roux
This script is regarding everything relating to the test data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import importing

def cubic(x,val1,val2):
    tck = interpolate.splrep(val1,val2)
    return interpolate.splev(x,tck)

def cubic_spline(x,val1,val2):
    cs = interpolate.CubicSpline(val1, val2)
    xvals = cs(x)
    return xvals

def make_P(data):
    '''Assume correct for now'''
    P1val = 1
    P2val = 3
    P3val = 0
    P4val = 2
    P5val = 4
    output = np.zeros((len(data),5))
    for j in range(len(data)):
        P1 = data[j,P1val]
        P2 = data[j,P2val]
        P3 = data[j,P3val]
        P4 = data[j,P4val]
        P5 = data[j,P5val]
        
        output[j,0] = P1
        output[j,1] = P2
        output[j,2] = P3
        output[j,3] = P4
        output[j,4] = P5
    return output

def insane_make_P(data,a,b,c,d,e):
    P1val = a
    P2val = b
    P3val = c
    P4val = d
    P5val = e
    output = np.zeros((len(data),5))
    for j in range(len(data)):
        P1 = data[j,P1val]
        P2 = data[j,P2val]
        P3 = data[j,P3val]
        P4 = data[j,P4val]
        P5 = data[j,P5val]
        
        output[j,0] = P1
        output[j,1] = P2
        output[j,2] = P3
        output[j,3] = P4
        output[j,4] = P5
    return output

def lets_not_go_mentally_insane(d1,d2,calibration):
    import itertools
    import math
    output = []
    looplist = list(itertools.permutations([0,1,2,3,4]))
    for k in looplist:
        val1 = k[0]
        val2 = k[1]
        val3 = k[2]
        val4 = k[3]
        val5 = k[4]
        merged_data = merge_data(insane_make_P(d1,val1,val2,val3,val4,val5),get_position_data(d2))
        results= make_big_array(merged_data, calibration)
        if not math.isnan(results[0,14]):
            output.append(k)
    return output

def get_position_data(data):
    '''
    Takes the raw output from labview and converts it into a [n,2] matrix

    Parameters
    ----------
    data : Raw Labview data
        Raw Data.

    Returns
    -------
    Matrix of positions in order of test.

    '''
    r,c = data.shape
    multiply = int(c/2)
    output = np.zeros((len(data)*multiply,2))
    counter = 0
    for k in range(multiply):
        for j in range(len(data)):
            Z = data[j,2*k+1]
            Y = data[j,2*k]
            output[counter,0] = Z
            output[counter,1] = Y
            counter += 1
    return output

def plot_position_data(data,c):
    plt.figure("Position Data")
    plt.scatter(data[:,0],data[:,1],c='black',s=10)
    plt.title("Coordinates at which values were measured",fontsize=28)
    plt.xlabel(r"$Z$ [mm]",fontsize=22)
    plt.ylabel(r"$Y$ [mm]",fontsize=22,rotation=0,labelpad=30)
    return 'Done'

def merge_data(pvals,position):
    output = np.zeros((len(pvals),7))
    for k in range(len(pvals)):
        output[k,0] = pvals[k,0] #P1
        output[k,1] = pvals[k,1] #P2
        output[k,2] = pvals[k,2] #P3
        output[k,3] = pvals[k,3] #P4
        output[k,4] = pvals[k,4] #P5
        output[k,5] = position[k,0] #Z
        output[k,6] = position[k,1] #Y
    return output


def make_pdash(coeffs,poff):
    pdash = np.zeros(len(coeffs))
    counter = 0
    for j in coeffs:
        pdash[counter] = 9.8*(j[3]+0.15)-poff
        counter += 1
    return pdash

def make_ydash(pdash,coeffs,calibration_data):
    def make_inter(data):
        output = np.zeros((len(data),2))
        for j in range(len(data)):
            output[j,0] = data[j,0] #Pitch
            output[j,1] = data[j,2] #Cpy
        return output
    counter = 0
    ydash = np.zeros(len(pdash))
    for j in pdash:
        Cpy_point = coeffs[counter,2] #Cpy point
        interpolation_point = [[j,Cpy_point]]
        interpolation_input = make_inter(calibration_data)
        '''Interpolate using j and Cpy to calculate ydash'''
        ydash[counter] = interpolate.griddata(interpolation_input, calibration_data[:,1], interpolation_point, method='cubic')
        counter += 1
    return ydash

def make_p(ydash,coeffs,calibration_data):
    def make_inter(data):
        output = np.zeros((len(data),2))
        for j in range(len(data)):
            output[j,0] = data[j,1] #Yaw 
            output[j,1] = data[j,3] #Cpp
        return output
    counter = 0
    pvals = np.zeros(len(ydash))
    for k in ydash:
        Cpp_point = coeffs[counter,3] #Cpp Point
        interpolation_point = [[k,Cpp_point]]
        interpolation_input = make_inter(calibration_data)
        pvals[counter] = interpolate.griddata(interpolation_input, calibration_data[:,0], interpolation_point,method='cubic')
        counter += 1
    return pvals

def correct_p(pvals,poff):
    for j in range(len(pvals)):
        pvals[j] = pvals[j]-poff
    return pvals

def get_Cpts(pfinal,yfinal,testdata,calibration_data):
    def make_inter(p,y):
        output = np.zeros((len(p),2))
        for j in range(len(p)):
            output[j,0] = p[j]
            output[j,1] = y[j]
        return output
    interpoints = make_inter(pfinal, yfinal)
    interpolation_input = calibration_data[:,0:2] #Pitch and Yaw
    Cpts_data = interpolate.griddata(interpolation_input, calibration_data[:,6], interpoints, method='cubic')
    return Cpts_data

def do_v_steps(testdataraw,positionraw,calibration_data,poff=2.5): #NEED Calibration DATA
    '''
    testdataraw = Labview output from five-hole probe (P1-P5)
    positionraw = Position matrix from Labview
    Calibration Data = Output excel from calibration.py 
    poff = some value (we can vary it and test the difference later)
    '''

    '''Step 1: Make coeffs'''
    posdata = get_position_data(positionraw)
    pvalues = make_P(testdataraw)
    combined = merge_data(pvalues, posdata)
    coeffs = new_coeffs(combined)
    '''Step 2: Generate Pdash'''
    pdash = make_pdash(coeffs, poff)
    '''Setp 3: Generate Ydash'''
    ydash = make_ydash(pdash, coeffs, calibration_data)
    '''Step 4: Get Pitch Angle'''
    pvals = make_p(ydash,coeffs,calibration_data)
    p_angle = correct_p(pvals,poff)
    '''Step 5: Get Yaw Angle'''
    y_angle = make_ydash(p_angle, coeffs, calibration_data)
    '''Step 6: Get Cpts '''
    Cpts = get_Cpts(p_angle, y_angle, coeffs, calibration_data)
    return Cpts, p_angle, y_angle

def new_coeffs(data):
    '''
    Takes the five Pressure values and position Z,Y.
    [P1--P5,Z,Y]

    Parameters
    ----------
    data : nx5 array of Pressure vals
        The pressure values recorded exlusively by the probe.

    Returns
    -------
    A array of the Cpp and Cpy vals with the associated positions -> [Z,Y,Cpy,Cpp]

    '''
    def calc_Pavg(datapoint):
        return (datapoint[1]+datapoint[2]+datapoint[3]+datapoint[4])/4
    
    def calc_Cpy(datapoint):
        P1 = datapoint[0]
        P2 = datapoint[1]
        P3 = datapoint[2]
        Pavg = calc_Pavg(datapoint)
        return (P3-P2)/(P1-Pavg)
    
    def calc_Cpp(datapoint):
        P1 = datapoint[0]
        P4 = datapoint[3]
        P5 = datapoint[4]
        Pavg = calc_Pavg(datapoint)
        return (P5-P4)/(P1-Pavg)
    
    output = np.zeros((len(data),4))
    for j in range(len(data)):
        Cpy = calc_Cpy(data[j])
        Cpp = calc_Cpp(data[j])
        output[j,2] = Cpy
        output[j,3] = Cpp
        output[j,0] = data[j,5] #Z
        output[j,1] = data[j,6] #Y
    return output

def sample_new_coeffs(data):
    def calc_Pavg(datapoint):
        return (datapoint[3]+datapoint[4]+datapoint[5]+datapoint[6])/4
    
    def calc_Cpy(datapoint):
        P1 = datapoint[2]
        P2 = datapoint[3]
        P3 = datapoint[4]
        Pavg = calc_Pavg(datapoint)
        return (P3-P2)/(P1-Pavg)
    
    def calc_Cpp(datapoint):
        P1 = datapoint[2]
        P4 = datapoint[5]
        P5 = datapoint[6]
        Pavg = calc_Pavg(datapoint)
        return (P5-P4)/(P1-Pavg)
    
    output = np.zeros((len(data),4))
    for j in range(len(data)):
        Cpy = calc_Cpy(data[j])
        Cpp = calc_Cpp(data[j])
        output[j,2] = Cpy
        output[j,3] = Cpp
        output[j,0] = data[j,0] #Z
        output[j,1] = data[j,1] #Y
    return output

def spatial_resolutionP2P3(data,d=0,c=30):
    '''First let us do P2 and P3'''
    index = np.lexsort((data[:,0],data[:,1])) # First Y and then Z
    for k in range(c):
        vals = np.zeros((int(len(data)/c),3))
        counter = 0
        for j in range(len(vals)):
            vals[counter,0] = data[index[k*int(len(data)/c)+counter],0] #Z
            vals[counter,1] = data[index[k*int(len(data)/c)+counter],3] #P2
            vals[counter,2] = data[index[k*int(len(data)/c)+counter],4] #P3
            # vals[counter,2] = data[index[k*c+counter],4] #P3
            counter += 1
        # print(len(vals))
        # print(vals)
        P2new = cubic(vals[:,0]-d,vals[:,0],vals[:,1])
        P3new = cubic(vals[:,0]+d,vals[:,0],vals[:,2])
        for j in range(len(P2new)):
            data[index[k*int(len(data)/c)+j],3] = P2new[j]
            data[index[k*int(len(data)/c)+j],4] = P3new[j]
    return data
    
def spatial_resolutionP4P5(data,d=0,c=30):
    index = np.lexsort((data[:,1],data[:,0]))
    for k in range(c):
        vals = np.zeros((int(len(data)/c),3))
        counter = 0
        for j in range(len(vals)):
            vals[counter,0] = data[index[k*int(len(data)/c)+counter],1] #Y
            vals[counter,1] = data[index[k*int(len(data)/c)+counter],5] #P4
            vals[counter,2] = data[index[k*int(len(data)/c)+counter],6] #P5
            counter += 1
        P4new = cubic(vals[:,0]-d,vals[:,0],vals[:,1])
        P5new = cubic(vals[:,0]+d,vals[:,0],vals[:,2])
        for j in range(len(P4new)):
            data[index[k*int(len(data)/c)+j],5] = P4new[j]
            data[index[k*int(len(data)/c)+j],6] = P5new[j]
    return data
        
# test_raw_data_sorted,v_const,Pd_const = importing.windtunnel_data()
# Pvals = make_P(test_raw_data_sorted)
# position_raw_data = importing.position_data()
# calibration_data = importing.coefficients()

def make_big_array(merged_data,calibration_data,a1=1,a2=3,a3=0,a4=2,a5=4,d=0,c=30):
    '''
    Makes a big array containing all our wanted outputs

    Parameters
    ----------
    merged_data : numpy array
        The combined position and test data.
    calibration_data : numpy array
        Calibration Data from the Calibration.py output excel file.
    d : float, optional
        The diameter of the five-hole probe. The default is 0.
    c : Int, optional
        Makes the process of spatial correction easier - should be equal to the amount of data collected per pass. The default is 30.

    Returns
    -------
    output : numpy array
        All our wanted outputs in one massive array.

    '''
    output = np.zeros((len(merged_data),18))
    for j in range(len(merged_data)):
        output[j,0] = merged_data[j,5] #Z
        output[j,1] = merged_data[j,6] #Y
        '''Check order'''
        output[j,2] = merged_data[j,a1] #P1
        output[j,3] = merged_data[j,a2] #P2
        output[j,4] = merged_data[j,a3] #P3
        output[j,5] = merged_data[j,a4] #P4
        output[j,6] = merged_data[j,a5] #P5
    '''Apply Spatial Correction before continuing'''
    '''----------------------------------------'''
    output = spatial_resolutionP2P3(output,d,c)
    output = spatial_resolutionP4P5(output,d,c)
    '''----------------------------------------'''
    for k in range(len(merged_data)):
        P1 = output[k,2] #P1
        P2 = output[k,3] #P2
        P3 = output[k,4] #P3
        P4 = output[k,5] #P4
        P5 = output[k,6] #P5
        Pavg = (output[k,3]+output[k,4]+output[k,5]+output[k,6])/4
        output[k,7] = (output[k,3]+output[k,4]+output[k,5]+output[k,6])/4 #Pavg
        '''SORT THIS OUT'''
        Ps = 86819
        # Ps = 65500
        Pt = Ps + (10.2)*100
        # Pt = Ps + (71.3)
        Cpy = -(P2-P3)/(P1-Pavg)
        Cpp = -(P4-P5)/(P1-Pavg)
        Cpt = (Pt-Pavg)/(P1-Pavg)
        Cps = (P1-Ps)/(P1-Pavg)
        output[k,8] = Pt
        output[k,9] = Ps
        output[k,10] = Cpy
        output[k,11] = Cpp
        output[k,12] = Cpt
        output[k,13] = Cps
    
    coeffs = output[:,8:12]
    poff = 2.5
    '''Step 2: Generate Pdash'''
    pdash = make_pdash(coeffs, poff)
    # print(pdash.shape)
    '''Setp 3: Generate Ydash'''
    ydash = make_ydash(pdash, coeffs, calibration_data)
    '''Step 4: Get Pitch Angle'''
    pvals = make_p(ydash,coeffs,calibration_data)
    p_angle = correct_p(pvals,poff)
    '''Step 5: Get Yaw Angle'''
    y_angle = make_ydash(p_angle, coeffs, calibration_data)
    '''Step 6: Get Cpts '''
    Cpts = get_Cpts(p_angle, y_angle, coeffs, calibration_data)
    rho = 1.225
    for k in range(len(merged_data)):
        output[k,14] = Cpts[k]
        output[k,15] =p_angle[k]
        output[k,16] = y_angle[k]
        V = np.sqrt(2*output[k,14]*np.abs(output[k,2]-output[k,7])/rho)
        output[k,17] = V
    return output

def sample_big_array(merged_data,calibration_data):
    '''Repeated the name 'Merge Data' because replacing it would take really long -- it should actually be collected data'''
    output = np.zeros((len(merged_data),18))
    for j in range(len(merged_data)):

        output[j,0] = merged_data[j,0] +140#Z
        output[j,1] = merged_data[j,1] #Y
        output[j,2] = merged_data[j,2] #P1
        output[j,3] = merged_data[j,3] #P2
        output[j,4] = merged_data[j,4] #P3
        output[j,5] = merged_data[j,5] #P4
        output[j,6] = merged_data[j,6] #P5
    '''-------------------'''
    output = spatial_resolutionP2P3(output,d=0,c=37)
    output = spatial_resolutionP4P5(output,d=0,c=35)
    '''-------------------'''
    for j in range(len(merged_data)):
        P1 = output[j,2]
        P2 = output[j,3]
        P3 = output[j,4]
        P4 = output[j,5]
        P5 = output[j,6]
        Pavg = (P2+P3+P4+P5)/4
        output[j,7] = Pavg #Pavg
        Pt = merged_data[j,7]
        output[j,8] = merged_data[j,7] #Pt
        Ps = merged_data[j,8]
        output[j,9] = merged_data[j,8] #Ps
        
        Cpy = -(P2-P3)/(P1-Pavg)
        Cpp = -(P4-P5)/(P1-Pavg)
        Cpt = (Pt-Pavg)/(P1-Pavg)
        Cps = (P1-Ps)/(P1-Pavg)
        # Cpts = (Pt-Ps)/(P1-Pavg)
        
        output[j,10] = Cpy
        output[j,11] = Cpp
        '''Review if necessary first'''
        output[j,12] = Cpt
        output[j,13] = Cps
        # output[j,14] = Cpts
    coeffs = output[:,8:12]
    poff = 4.5
    '''Step 2: Generate Pdash'''
    pdash = make_pdash(coeffs, poff)
    # print(pdash.shape)
    '''Setp 3: Generate Ydash'''
    ydash = make_ydash(pdash, coeffs, calibration_data)
    '''Step 4: Get Pitch Angle'''
    pvals = make_p(ydash,coeffs,calibration_data)
    p_angle = correct_p(pvals,poff)
    '''Step 5: Get Yaw Angle'''
    y_angle = make_ydash(p_angle, coeffs, calibration_data)
    '''Step 6: Get Cpts '''
    Cpts = get_Cpts(p_angle, y_angle, coeffs, calibration_data)
    rho = 1.225
    for k in range(len(merged_data)):
        output[k,14] = Cpts[k]
        output[k,15] =p_angle[k]
        output[k,16] = y_angle[k]
        V = np.sqrt(2*output[k,14]*np.abs(output[k,2]-output[k,7])/rho)
        output[k,17] = V
    return output

def make_SAMPLE_DATA_array(merged_data,calibration_data,a1=1,a2=3,a3=0,a4=2,a5=4,d=0,c=30):
    output = np.zeros((len(merged_data),18))
    for j in range(len(merged_data)):
        output[j,0] = merged_data[j,5] #Z
        output[j,1] = merged_data[j,6] #Y
        '''Check order'''
        output[j,2] = merged_data[j,a1] #P1
        output[j,3] = merged_data[j,a2] #P2
        output[j,4] = merged_data[j,a3] #P3
        output[j,5] = merged_data[j,a4] #P4
        output[j,6] = merged_data[j,a5] #P5
        output[j,7] = merged_data[j,7] #Pavg
        output[j,8] = merged_data[j,8] #Pt
        output[j,9] = merged_data[j,9] #Ps
    '''Apply Spatial Correction before continuing'''
    '''----------------------------------------'''
    output = spatial_resolutionP2P3(output,d,c=29)
    output = spatial_resolutionP4P5(output,d,c=59)
    '''----------------------------------------'''
    for k in range(len(merged_data)):
        P1 = output[k,2] #P1
        P2 = output[k,3] #P2
        P3 = output[k,4] #P3
        P4 = output[k,5] #P4
        P5 = output[k,6] #P5
        Pavg = output[k,7] #Pavg
        Ps = output[k,9] #Ps
        Pt = output[k,8] #Pt
        Cpy = -(P2-P3)/(P1-Pavg)
        Cpp = -(P4-P5)/(P1-Pavg)
        Cpt = (Pt-Pavg)/(P1-Pavg)
        Cps = (P1-Ps)/(P1-Pavg)

        output[k,10] = Cpy
        output[k,11] = Cpp
        output[k,12] = Cpt
        output[k,13] = Cps
    
    coeffs = output[:,8:12]
    poff = 2.5
    '''Step 2: Generate Pdash'''
    pdash = make_pdash(coeffs, poff)
    # print(pdash.shape)
    '''Setp 3: Generate Ydash'''
    ydash = make_ydash(pdash, coeffs, calibration_data)
    '''Step 4: Get Pitch Angle'''
    pvals = make_p(ydash,coeffs,calibration_data)
    p_angle = correct_p(pvals,poff)
    '''Step 5: Get Yaw Angle'''
    y_angle = make_ydash(p_angle, coeffs, calibration_data)
    '''Step 6: Get Cpts '''
    Cpts = get_Cpts(p_angle, y_angle, coeffs, calibration_data)
    rho = 1.17
    for k in range(len(merged_data)):
        output[k,14] = Cpts[k]
        output[k,15] =p_angle[k]
        output[k,16] = y_angle[k]
        V = np.sqrt(2*output[k,14]*np.abs(output[k,2]-output[k,7])/rho)
        output[k,17] = V
    return output

def do_test():
    wind, pos, coeff = importing.testing()
    test_raw_data_sorted = wind
    position_raw_data = pos
    calibration_data = coeff
    return test_raw_data_sorted, position_raw_data, calibration_data

# test_raw_data_sorted, position_raw_data, calibration_data = do_test()

def use_sample_data():
    collected = importing.sample_data('SAMPLE: Collected')
    calibration = importing.sample_data('SAMPLE: Calibration')
    return collected, calibration

def pressure_plot_for_report(pvals):
    fig, ax = plt.subplots(figsize=(5,5))
    
    c = 30
    
    line, = ax.plot(pvals[:,0],label="P1")
    xout = []
    for k in range(int(len(pvals)/c)):
        minval_x = pvals[c*k:c*k+c,0].argmin()
        ax.scatter(k*c+minval_x,pvals[k*c+minval_x,0])
        xout.append(k*c+minval_x)
        ax.annotate(str(k*c+minval_x),xy=(k*c+minval_x,pvals[k*c+minval_x,0]-4),xycoords='data')
        
    xvals = np.array(xout)
    avg = []
    for j in range(len(xvals)-1):
        diff = xvals[j+1]- xvals[j]
        avg.append(diff)
    avg = np.array(avg).mean()
    ax.annotate(r'Average distance between minimums, $\bar{x}_{min}$ ='+'{}'.format(avg),xy=(600,260),xycoords='data')
    # plt.figure('Pressure')
    # plt.plot(pvals[:,0],label="P1")
    # plt.plot(pvals[:,1],label="P2")
    # plt.plot(pvals[:,2],label="P3")
    # plt.plot(pvals[:,3],label="P4")
    # plt.plot(pvals[:,4],label="P5")
    # plt.legend()
    # '''Add pointers'''
    # line, = ax.plot(pvals[:,1],label="P2")
    # line, = ax.plot(pvals[:,2],label="P3")
    # line, = ax.plot(pvals[:,3],label="P4")
    # line, = ax.plot(pvals[:,4],label="P5")
    
    return avg
    
def get_Velocity(Cpts,testdataraw,rho=1.225):
    '''
    Testdata raw will be used once again for this - use make_P to get it into the right shape

    Parameters
    ----------
    Cpts : TYPE
        DESCRIPTION.
    testdata : TYPE
        DESCRIPTION.
    rho : TYPE, optional
        DESCRIPTION. The default is 1.225.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''
    pvals = make_P(testdataraw)
    output = np.zeros(len(Cpts))
    for j in range(len(Cpts)):
        pavg = (pvals[j,1]+pvals[j,2]+pvals[j,3]+pvals[j,4])/4
        output[j] = np.sqrt((2*Cpts[j]*np.abs(pvals[j,0]-pavg))/rho)
    return output

def sample_get_Velocity(Cpts,collected,rho=1.225):
    output = np.zeros(len(Cpts))
    for j in range(len(Cpts)):
        output[j] = np.sqrt((2*Cpts[j]*np.abs(collected[j,2]-collected[j,7]))/rho)
    return output

def make_velocity_components(V,p,y):
    Vt = np.zeros(len(V))
    Vr = np.zeros(len(V))
    Vz = np.zeros(len(V))
    for j in range(len(V)):
        Vt[j] = V[j]*np.cos(y[j])*np.cos(p[j])
        Vr[j] = V[j]*np.sin(y[j])
        Vz[j] = V[j]*np.cos(y[j])*np.sin(p[j])
    return Vt,Vr,Vz

def get_Velocity_big(bigboy):
    output = np.zeros((len(bigboy),5)) ###Seems this one might be wrong###
    for j in range(len(bigboy)):
        output[j,0] = bigboy[j,0] #Z
        output[j,1] = bigboy[j,1] #Y
        V = bigboy[j,17] 
        p = bigboy[j,15] * np.pi/180
        y = bigboy[j,16] * np.pi/180
        
        Vr = V*np.cos(y)*np.cos(p)
        Vt = V*np.sin(y)
        Vz = V*np.cos(y)*np.sin(p)
        
        Vx = Vr*np.cos(Vt)
        Vy = Vr*np.sin(Vt)
        # Vz = Vz
        output[j,2] = Vx
        output[j,3] = Vy
        output[j,4] = Vz
    return output

def get_Velocity_big_alternate(bigboy):
    output = np.zeros((len(bigboy),5))
    for j in range(len(bigboy)):
        output[j,0] = bigboy[j,0] #Z
        output[j,1] = bigboy[j,1] #Y
        V = bigboy[j,17] 
        p = bigboy[j,15] * np.pi/180
        y = bigboy[j,16] * np.pi/180
        
        Vr = V*np.cos(y)*np.cos(p)
        Vt = V*np.sin(y)
        Vz = V*np.cos(y)*np.sin(p)
        
        Vx = Vr
        Vy = Vt
        # Vz = Vz
        output[j,2] = Vx
        output[j,3] = Vy
        output[j,4] = Vz
    return output

def get_Velocity_Jono_version(bigboy, v=37):
    output = np.zeros((len(bigboy),5))
    for j in range(len(bigboy)):
        output[j,0] = bigboy[j,0] #Z
        output[j,1] = bigboy[j,1] #Y
        # V = bigboy[j,17] 
        p = bigboy[j,15] * np.pi/180
        y = bigboy[j,16] * np.pi/180
        
        Vr = v*np.cos(y)*np.cos(p)
        Vt = v*np.sin(y)
        Vz = v*np.cos(y)*np.sin(p)
        
        Vx = Vr
        Vy = Vt
        # Vz = Vz
        output[j,2] = Vx
        output[j,3] = Vy
        output[j,4] = Vz
    return output

def downwash_correction(Vdata,d=3.2/1000,c=30):
    output = np.zeros((len(Vdata),5))
    # temp = np.zeros((len(Vdata),5))
    def first_derivative(Va,Vb,Z1,Z2):
        returnval = (Va-Vb)/(Z1-Z2)
        return returnval
    # Vz first
    index = np.lexsort((Vdata[:,0],Vdata[:,1])) # First Y and then Z
    counter = 0
    delD = 0.2*d
    for j in range(c):
        for k in range(c-1):
            Vcurrent = Vdata[index[j*30+k],2] #Vx current
            Vnext = Vdata[index[j*30+k+1],2] #Vx next
            Zcurrent = Vdata[index[j*30+k],0] #Zval
            Znext = Vdata[index[j*30+k+1],0] #Znext
            val = first_derivative(Vcurrent, Vnext, Zcurrent, Znext) #f'
            output[index[j*30+k],4] = Vdata[index[j*30+k],4]+delD*val #Vz
            output[index[j*30+k],0] = Zcurrent #Z
            output[index[j*30+k],2] = Vcurrent #Vx value does not need to change
        output[index[j*30+29],4] = Vdata[index[j*30+29],4] #New Vz
        output[index[j*30+29],0] = Zcurrent #Z
        output[index[j*30+29],2] = Vcurrent #Vx value does not need to change
        
    index2 = np.lexsort((Vdata[:,1],Vdata[:,0])) #First Z and then Y
    for l in range(c):
        for s in range(c-1):
            Vcurrent = Vdata[index2[l*30+s],2] #Vx current
            Vnext = Vdata[index2[l*30+s+1],2] #Vx next 
            Ycurrent = Vdata[index2[l*30+s],1] #Current Y val
            Ynext = Vdata[index2[l*30+s+1],1] #Next Y val
            val = first_derivative(Vcurrent, Vnext, Ycurrent, Ynext)
            output[index2[l*30+s],3] = Vdata[index2[l*30+s],3] + delD*val #New Vy
            output[index2[l*30+s],1] = Ycurrent #Y
        output[index2[l*30+29],3] = Vdata[index2[l*30+29],3]
        output[index2[l*30+29],1] = Ycurrent #Y
    return output

def use_seaborn(vold,vnew):
    import seaborn as sns
    Vz_diff = np.zeros(len(vold))
    Vy_diff = np.zeros(len(vold))
    for j in range(len(vold)):
        vyu = vold[j,3]
        vy = vnew[j,3]
        Vy_diff[j] = vy-vyu #not using abs
        
        vzu = vold[j,4]
        vz = vnew[j,4]
        Vz_diff[j] = vz-vzu
    
    fig = plt.figure("Vy")
    sns.distplot(Vy_diff, label=r"$V_y$")
    plt.title(r"Distribution of the changes between original and downwash corrected $V_y$",fontsize=28)
    plt.xlabel("Value Difference",fontsize=26)
    plt.ylabel("Frequency",fontsize=26)
    plt.legend(fontsize=22)
    
    fig = plt.figure("Vz")
    sns.distplot(Vz_diff, label=r"$V_z$")
    plt.title(r"Distribution of the changes between original and downwash corrected $V_z$",fontsize=28)
    plt.xlabel("Value Difference",fontsize=26)
    plt.ylabel("Frequency",fontsize=26)
    plt.legend(fontsize=22)
    return None

def do_all(merged_data,calibration_data):
    bigboy = make_big_array(merged_data, calibration_data)
    V_data = get_Velocity_big(bigboy)
    return bigboy, V_data

def show_V(Vin):
    '''

    Parameters
    ----------
    Vin : Numpy Array
        Z,Y,V1,V2,V3.

    Returns
    -------
    None - only makes a plot of the data

    '''
    Z = Vin[:,0]
    Y = Vin[:,1]
    # V1 = Vin[:,2]
    u = Vin[:,2]-39.0
    v = Vin[:,3]
    w = Vin[:,4]
    # V2 = Vin[:,3]
    # V3 = Vin[:,4]
    fig = plt.figure()
    plt.quiver(Z, Y,w,v,scale_units='xy')
    plt.show()
    return None

def try_3D_plot(Vin):
    from mpl_toolkits.mplot3d import Axes3D
    X = np.zeros(len(Vin))
    for k in range(len(Vin)):
        X[k] = 1
    Y = Vin[:,1]
    Z = Vin[:,0]
    u = Vin[:,2]
    for j in range(len(u)):
        u[j] = u[j] - 37
    v = Vin[:,3]
    w = Vin[:,4]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.quiver(X,Y,Z,u,v,w)
    plt.show()
    return None

def basic_quiver(data):
    '''
    PLots a simple quiver plot

    Parameters
    ----------
    data : Array
        In the form: Z, Y, Vx, Vy, Vz.

    Returns
    -------
    None.

    '''
    plt.figure("Quiver PLot")
    Z = data[:,0]
    Y = data[:,1]
    u = data[:,2] #Vx
    v = data[:,3] #Vy
    w = data[:,4] #Vz
    plt.quiver(Z,Y,w,v,scale_units='xy',linewidth=0.00002, width=0.0008)
    
    plt.figure('Vx')
    plt.plot(u,label="Vx")
    plt.legend()
    
    plt.show()