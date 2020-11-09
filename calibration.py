# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:32:01 2020

@author: Francois le Roux
This python script is to be used for all processing to be done regarding calibration of the five-hole probe.
"""
import importing
import numpy as np
import matplotlib.pyplot as plt
'''Assume data coming in is in the format: Pitch, Yaw, P1,P2,P3,P4,P5,Pt,Ps'''

#calibration_raw_sorted = importing.calibration_data()

# Pd = P #Dynamic Pressure - delta P
# Ps = 87000 #According to Google - Constant
# Pt = Ps+Pd

def make_dict45(data):
    def gen_dict(data):
        out = {'Y':data[0],'P1':data[1],'P2':data[3],'P3':data[4],'P4':data[5],'P5':data[6]}
        return out
    output = {}
    counter = 0
    for j in data:
        if j[0] not in output:
            output[j[0]] = {counter : gen_dict(j)}
        else:
            output[j[0]].update({counter: gen_dict(j)})
        counter += 1
    return output

def generate_calibration_data(raw_input):
    '''Only if the excel file with the probe calibration data is not already in existance'''
    #MAKE SURE: --> RAW = Pitch,Yaw,P1,P2,P3,P4,P5,Pt,Ps
    import xlsxwriter
    output = np.zeros((len(raw_input),7))
    for j in range(len(output)):
        Pitch = raw_input[j,0]
        Yaw = raw_input[j,1]
        P1 = raw_input[j,2]
        P2 =raw_input[j,3]
        P3 =raw_input[j,4]
        P4 =raw_input[j,5]
        P5 =raw_input[j,6]
        Pt =raw_input[j,7]
        Ps =raw_input[j,8]
        Pavg = (P2+P3+P4+P5)/4
        
        Cpy = -(P2-P3)/(P1-Pavg)
        Cpp = -(P4-P5)/(P1-Pavg)
        Cpt = (Pt-Pavg)/(P1-Pavg)
        Cps = (P1-Ps)/(P1-Pavg)
        Cpts = (Pt-Ps)/(P1-Pavg)
        
        output[j,0] = Pitch
        output[j,1] = Yaw
        output[j,2] = Cpy
        output[j,3] = Cpp
        output[j,4] = Cpt
        output[j,5] = Cps
        output[j,6] = Cpts
    def generate_workbook(Final_Data,name):
    # index = np.lexsort((Final_Data[:,1],-Final_Data[:,0]))
        workbook = xlsxwriter.Workbook(name+".xlsx")
        worksheet = workbook.add_worksheet("Data")
        row = 0
        col = 0
        # worksheet.write(row,col,"Pitch [mm]")
        # worksheet.write(row,col+1,"Yaw [mm]")
        # worksheet.write(row,col+2,"Cpy")
        # worksheet.write(row,col+3,"Cpp")
        # worksheet.write(row,col+4,"Cpt")
        # worksheet.write(row,col+5,"Cps")
        # worksheet.write(row,col+6,"Cpts")
        # row += 1
        for x in range(len(Final_Data)):
            worksheet.write(row,col,Final_Data[x,0])
            worksheet.write(row,col+1,Final_Data[x,1])
            worksheet.write(row,col+2,Final_Data[x,2])
            worksheet.write(row,col+3,Final_Data[x,3])
            worksheet.write(row,col+4,Final_Data[x,4])
            worksheet.write(row,col+5,Final_Data[x,5])
            worksheet.write(row,col+6,Final_Data[x,6])
            row += 1
        workbook.close()
    name = input("Please enter a name for the workbook")
    generate_workbook(output, name)
    return 'Complete'

def plotting(p_sorted, y_sorted, num1= 1, num2=2):
    '''p_sorted refers to a dictionary of values sorted according to Pitch, and similarly for the y_sorted dictionary is Yaw'''
    def gen_details(num):
        '''Number is 0 = Pitch, 1 = Yaw, 2 = Cpy, 3= Cpp, 5 = Cps, 4= Cpt, 6 = Cpts Returns name, description'''
        if num == 3:
            name = 'Cpp'
            description = r'$C_{pp}$'
            indexval = 3
        elif num == 5:
            name = 'Cps'
            description = r'$C_{ps}$'
            indexval = 5
        elif num == 4:
            name = 'Cpt'
            description = r'$C_{pt}$'
            indexval = 4
        elif num == 6:
            name = 'Cpts'
            description = r'$C_{pts}$'
            indexval = 6
        elif num == 0:
            name = 'Pitch'
            description = r'Pitch ($\alpha$)'
            indexval = 0
        elif num == 1:
            name = 'Yaw'
            description = r'Yaw ($\beta$)'
            indexval = 1
        else:
            name = 'Cpy'
            description = r'$C_{py}$'
            indexval = 2
        return name, description, indexval
    
    details_x = gen_details(num1)
    details_y = gen_details(num2)
    if details_x[0] == 'Yaw': #Plots Yaw versus any coefficient
        for k in p_sorted.keys():
            x = []
            y = []
            for j in p_sorted[k].keys():
                vals = p_sorted[k][j]
                x.append(vals[details_x[0]])
                y.append(vals[details_y[0]])
            plt.scatter(x,y,label='Pitch = '+str(k)+r'$\degree$')
            plt.plot(x,y)
    elif details_x[0] == 'Pitch': #Plots Pitch versus any coefficient
        for k in y_sorted.keys():
            x = []
            y = []
            for j in y_sorted[k].keys():
                vals= y_sorted[k][j]
                x.append(vals[details_x[0]])
                y.append(vals[details_y[0]])
            plt.scatter(x,y,label="Yaw = "+str(k)+r'$\degree$')
            plt.plot(x,y)
    else: #Carpet Plot
        for k in p_sorted.keys():
            x = []
            y = []
            for j in p_sorted[k].keys():
                vals = p_sorted[k][j]
                x.append(vals[details_x[0]])
                y.append(vals[details_y[0]])
            plt.scatter(x,y,label='Pitch = ' +str(k)+r'$\degree$')
            plt.plot(x,y)
        for k in y_sorted.keys():
            x = []
            y = []
            for j in y_sorted[k].keys():
                vals = y_sorted[k][j]
                x.append(vals[details_x[0]])
                y.append(vals[details_y[0]])
            plt.scatter(x,y,label="Yaw = "+str(k)+r'$\degree$')
            plt.plot(x,y)
            
    plt.title(details_y[1]+' vs '+details_x[1],fontsize=28)
    plt.xlabel(details_x[1],fontsize=28)
    plt.ylabel(details_y[1],fontsize=28)
    plt.legend()
    plt.grid(True,which='both')
    plt.show()
    return None #p_sorted, y_sorted

def make_py_dicts(data_in):
    def make_dict_y(data):
        index = np.lexsort((data[:,0],data[:,1]))
        def gen_dict(data):
            out = {'Pitch':data[0],'Cpy':data[2],'Cpp':data[3],'Cpt':data[4],'Cps':data[5],'Cpts':data[6]}
            return out
        output = {}
        counter = 0
        for k in data:
            j = data[index[counter]]
            if j[1] not in output:
                output[j[1]] = {counter: gen_dict(j)}
            else:
                output[j[1]].update({counter: gen_dict(j)})
            counter += 1
        return output
        
    def make_dict_p(data):
        index = np.lexsort((data[:,1],data[:,0]))
        def gen_dict(data):
            out = {'Yaw':data[1],'Cpy':data[2],'Cpp':data[3],'Cpt':data[4],'Cps':data[5],'Cpts':data[6]}
            return out
        output = {}
        counter = 0
        for k in data:
            j = data[index[counter]]
            if j[0] not in output:
                output[j[0]] = {counter: gen_dict(j)}
            else:
                output[j[0]].update({counter: gen_dict(j)})
            counter += 1
        return output
    p = make_dict_p(data_in)
    y = make_dict_y(data_in)
    return p,y


# p_sorted, y_sorted = make_py_dicts(importing.coefficients())
# p_sorted, y_sorted = make_py_dicts(test)

# def plot_all(p_sorted=p_sorted,y_sorted=y_sorted):
def plot_all():
    p_sorted, y_sorted = make_py_dicts(importing.coefficients())
    '''Plots all calibration coefficients plots '''
    counter = 0
    for j in range(2): #Generate all plots
        for k in range(5):
            plt.figure(str(counter))
            plotting(p_sorted,y_sorted,num1=j,num2=k+2)
            counter += 1
    plt.figure(str(counter))
    plotting(p_sorted,y_sorted,num1=2,num2=3) #Carpet Plot
    return None

def plot_single(p_sorted,y_sorted,num1=1,num2=2):
    plotting(p_sorted,y_sorted,num1,num2)
