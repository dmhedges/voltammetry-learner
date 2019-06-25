# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:31:40 2018

@author: dmhedges
"""

import numpy as np
from nptdms import TdmsFile
from scipy import signal
#from peakutils.peak import indexes
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

huge_spon = '00_Nac1p_5_huge spons.tdms'
spon = '00_Nac1p_9_spons.tdms'

def read_tdms(file_name):
    """This function reads in a color plot tdms file from Demon Voltammetry export options and returns the 
    color plot as a 2-D Numpy Array with background subtracted. The background is the first CV.

    X-axis of array is time, Y-axis is an individual command voltage recording.

    Keyword Arg is the tdms file that you wish to analyze. 
    """

    # Access file as object
    tdms_file = TdmsFile(file_name)
    root_object = tdms_file.object()

    # extracting the number of collections in a recording from the tdms file properties
    num_collections = (root_object.property("Collection Duration (s)")*root_object.property("Collection Frequency (Hz)"))
    num_collections

    # Initializing list
    color_plot_lists = []

    # Populating the color plot lists with all of the collections from the recording
    for i in range(0,int(num_collections)):
        command_voltage = tdms_file.object('Data1', '%s' %i)
        command_voltage = command_voltage.data
        color_plot_lists.append(command_voltage)

    # Transforming the list of lists into a NumPy 2-D array and transposing it so time is the x-axis
    color_plot_array = np.array(color_plot_lists).transpose()
    # Check to make sure that the first column is a CV
    #plt.plot(color_plot_array[:,0])

    # Perform background subtraction
    back_sub_array = []
    initial_cv = color_plot_array[:,0]
    
    for column in color_plot_array.transpose():
        back_sub_array.append(column - initial_cv)
        
    color_plot_array = np.array(back_sub_array).transpose()
    
    
    # Apply Butterworth Filter
    b,a = signal.butter(4, 0.03, analog=False)
    butter = []
    for column in color_plot_array.transpose():
        #filtered = signal.filtfilt(b, a, column)
        butter.append(signal.filtfilt(b, a, column))
    final_color_plot_array = np.array(butter).transpose()
    
    return final_color_plot_array


def get_peak_indices(color_plot, threshold=0.5, neighborhood_size=10):
    """
    This function reads in a colormap (for example, the output of read_tdms()),
    and returns two lists for the x and y (respectively) coordinates for all local
    maxima.
    
    Keyword Args:
    color_plot is the output of the read_tdms function.
    threshold is the minimum size of the peak. Default is 2.
    neighborhood_size is how big of an area to look at the peak. Default is 40.
    """

    #from scipy.ndimage.filters import maximum_filter
    
    data_max = filters.maximum_filter(color_plot, neighborhood_size)
    maxima = (color_plot == data_max)
    data_min = filters.minimum_filter(color_plot, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
        
    return x, y


def plot_heat_peaks(filename, threshold=0.5, neighborhood_size=10):
    color_plot = read_tdms(filename)
    x,y = get_peak_indices(color_plot=color_plot, threshold=threshold, neighborhood_size=neighborhood_size)
    
    sns.set(rc={'figure.figsize': (10,7)})
    ax = sns.heatmap(color_plot, xticklabels=50, yticklabels=200, cmap='YlGn')
    plt.plot(x,y, 'ro')
    
    
def filter_by_ox_index(filename, threshold=0.5, neighborhood_size=10, lower=150, upper=500):
    color_plot = read_tdms(filename)
    x,y = get_peak_indices(color_plot=color_plot, threshold=threshold, neighborhood_size=neighborhood_size)

    x_new, y_new = [], []

    for index, i in enumerate(y):
        if i >= lower and i <= upper:
            x_new.append(x[index])
            y_new.append(y[index])
    
    return x_new,y_new

def preprocessing(filename, threshold=0.5, neighborhood_size=40, plot_me=False, lower=150, upper=500):
    tdms_file = read_tdms(filename)
    x,y = get_peak_indices(color_plot=tdms_file, threshold=threshold, neighborhood_size=neighborhood_size)
    
    if plot_me == True:
        plot_heat_peaks(filename, threshold=threshold, neighborhood_size=neighborhood_size)
    
    x_filtered, y_filtered = filter_by_ox_index(filename, threshold=threshold, neighborhood_size=neighborhood_size,
                                                lower=lower, upper=upper)
    
    return x_filtered,y_filtered