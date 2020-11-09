# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:30:40 2020

@author: Francois le Roux
This Python script is to be used in conjunction with the main program for all importing purposes of raw data (as output by LabView)
"""
import pandas as pd
import numpy as np

class ImportData():
    '''This class is the one to be used for the importing of data - outputs it as a numpy array'''
    def __init__(self,filename):
        self.filename = filename
        if filename == 'Calibration Coefficients':
            x = 0
        else:
            x = None
        df = pd.read_excel(f'{filename}',header=x)
        self.data = np.array(df)
    
    def get_data(self):
        return self.data
    

def prompt_data(info):
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox
    from tkinter.messagebox import showerror
    root = tk.Tk()
    messagebox.showinfo("Take Note:",f"Please select {info} data")
    # Button(root,text="Close",command=root.destroy).pack()
    file_path = filedialog.askopenfilename()
    if not file_path:
        showerror("Error","You did not select a file, or the file you selected is corrupted")
    root.destroy()
    root.mainloop()
    return file_path

def calibration_data_mirror():
    path = prompt_data("Calibration")
    # path = 'D:/Tuks/2020/MRN/422 - Progress/Windtunnel Data/Calibration-Clean.xlsx'
    data = ImportData(path).get_data()
    # v = input("Please enter the velocity (assumed constant) at which the test was conducted in m/s")
    # P = input("Please enter the static pressure value at which the test was conducted in Pa")
    v = (36.5+34.4)/2
    P = ((7.9+7.0)/2)*100
    
    Pd = P
    Ps = 86819
    Pt = Pd+Ps
    final_length = 0
    
    constant = Ps
    for k in range(len(data)):
        if np.round(data[k,7]) == 0:
            final_length += 1
    output = np.zeros((2*len(data)-final_length,9))
    for j in range(len(output)):
        if j < len(data): #(0-66)
            output[j,0] = data[j,7] #Pitch
            output[j,1] = data[j,6] #Yaw
            output[j,2] = data[j,1] +constant#P1
            output[j,3] = data[j,4]+constant #P2
            output[j,4] = data[j,5]+constant #P3
            output[j,5] = data[j,3]+constant #P4
            output[j,6] = data[j,2]+constant #P5

        else: #(67-121)
            countval = j-len(data)+final_length
            output[j,0] = -1*data[countval,7] #Mirrored Pitch
            output[j,1] = data[countval,6] #Yaw
            output[j,2] = data[countval,1]+constant #+np.random.randint(0,50)#P1
            output[j,3] = data[countval,4]+constant #+np.random.randint(0,50)#P2
            output[j,4] = data[countval,5]+constant #+np.random.randint(0,50)#P3
            output[j,5] = data[countval,2]+constant #+np.random.randint(0,50)#P4
            output[j,6] = data[countval,3]+constant #+np.random.randint(0,50)#P5
            '''I changed P4 and P5 around -- since its "swopped"'''
        output[j,7] = Pt
        output[j,8] = Ps
    return output

def calibration_data():
    path = prompt_data("Calibration")
    # path = 'D:/Tuks/2020/MRN/422 - Progress/Windtunnel Data/Calibration-Clean.xlsx'
    data = ImportData(path).get_data()
    # v = input("Please enter the velocity (assumed constant) at which the test was conducted in m/s")
    # P = input("Please enter the static pressure value at which the test was conducted in Pa")
    v = (41.7+41.4)/2
    P = (10.2)*100
    
    Pd = P
    Ps = 86819
    Pt = Pd+Ps
    constant = Ps
    output = np.zeros((len(data),9))
    for j in range(len(output)):
        output[j,0] = data[j,7] #Pitch
        output[j,1] = data[j,6] #Yaw
        output[j,2] = data[j,1] +constant#P1
        output[j,3] = data[j,4]+constant #P2
        output[j,4] = data[j,5]+constant #P3
        output[j,5] = data[j,3]+constant #P4
        output[j,6] = data[j,2]+constant #P5
        output[j,7] = Pt
        output[j,8] = Ps
    return output

def windtunnel_data():
    data = ImportData(prompt_data("WindTunnel")).get_data()
    v = input("Please enter the velocity (assumed constant) at which the test was conducted in m/s")
    P = input("Please enter the static pressure value at which the test was conducted in Pa")
    return data, v, P

def position_data():
    data = ImportData(prompt_data("Positioning")).get_data()
    return data

def coefficients():
    data = ImportData(prompt_data("Calibration Coefficients")).get_data()
    return data

def testing():
    wind = ImportData(prompt_data("WindTunnel")).get_data()
    pos = ImportData(prompt_data("Positioning")).get_data()
    coeff = ImportData(prompt_data("Calibration Coefficients")).get_data()
    return wind, pos, coeff

def sample_data(name):
    out = ImportData(prompt_data(name)).get_data()
    return out