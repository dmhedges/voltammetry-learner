# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:26:15 2019

@author: dmhedges
"""

import numpy as np
from numpy import diff
from numpy import trapz

from nptdms import TdmsFile

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy import stats
from scipy import signal
from scipy.signal import argrelmin
from scipy.signal import argrelmax

import seaborn as sns
import matplotlib.pyplot as plt


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

    # Populating the color plot lists with all of the collections from the recording
    color_plot_lists = []
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
        butter.append(signal.filtfilt(b, a, column))
    final_color_plot_array = np.array(butter).transpose()
    
    return final_color_plot_array


def get_peak_indices(color_plot, threshold=0.05, neighborhood_size=10):
    """
    This function reads in a colormap (for example, the output of read_tdms()),
    and returns two lists for the x and y (respectively) coordinates for all local
    maxima.
    
    Keyword Args:
    color_plot is the output of the read_tdms function.
    threshold is the minimum size of the peak. Default is 2.
    neighborhood_size is how big of an area to look at the peak. Default is 40.
    """
    
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


def plot_heat_peaks(color_plot, x, y):
    sns.set(rc={'figure.figsize': (10,7)})
    plt.plot(x,y, 'ro')


def filter_by_ox_index(color_plot, x, y, lower=150, upper=500):
    x_new, y_new = [], []

    for index, i in enumerate(y):
        if i >= lower and i <= upper:
            x_new.append(x[index])
            y_new.append(y[index])
    
    return x_new,y_new


def preprocessing(color_plot, x, y, threshold=0.5, neighborhood_size=40, plot_me=False, lower=150, upper=500):
    '''
    This function works all the previous functions in preprocessing together and returns the coordinates
    of the peaks
    '''

    if plot_me == True:
        plot_heat_peaks(color_plot, threshold=threshold, neighborhood_size=neighborhood_size)
    
    x_filtered, y_filtered = filter_by_ox_index(color_plot, x, y, lower=lower, upper=upper)
    
    return x_filtered,y_filtered


def get_cvs(color_plot, x_color, y_color):
    """
    It reads in a tdms file object and returns a list of the CVs in the same order as the x array from 
    preprocessing.get_peak_indices.
    """
    x = [int(i) for i in x_color]
    return [color_plot[:,i] for i in x]


def get_r2(cvs, volt):
    """
    This function takes the output from get_cvs and a sample voltammogram.
    
    Returns a list of r-squared values for the CVs given in the same order as the x values from
    preprocessing.read_tdms.
    """
    r2 = []
    for i in cvs:
        slope, intercept, r_value, p_value, std_err = stats.linregress(volt,i)
        r2.append(r_value**2)
    
    return r2


def get_rise_times(color_plot, x_color, y_color, collection_rate=1):
    """
    This function reads in a tdms file and returns a list of rise times for the peaks in the 
    same order as the x values from preprocessing.read_tdms.
    """

    x_int = [int(i) for i in x_color]
    y_int = [int(i) for i in y_color]
    
    # Take derivatives of IvsTs
    peak_and_times = []
    for idx,i in enumerate(y_int):
        ivst = color_plot[i]
        dy_1 = diff(ivst)/collection_rate
        dy_2 = diff(dy_1)/collection_rate

        peak = (x_int[idx], argrelmax(dy_2)[0])
        peak_and_times.append(peak)
        
    peak_with_time = []
    for i in peak_and_times:
        greater = []
        smaller = []
        equal = []
        for idx,time in enumerate(i[1]):
            if time < i[0]:
                smaller.append(time)
            #if time == i[0]:
            equal.append(time)
            if time > i[0]:
                greater.append(time)
        if len(smaller) != 0:
            start_time = smaller[-1]
        if len(smaller) == 0:
            start_time = equal[0]
        peak_with_time.append((i[0], start_time))
    
    rise_times = [i[0]-i[1] for i in peak_with_time]
    rise_times = [i/10 for i in rise_times]
    
    return np.absolute(rise_times)


def get_oxid_idx(x_color, y_color):
    """
    This function reads in a tdms file and returns a list of the oxidation indices for the peaks in the 
    same order as the x values from preprocessing.read_tdms.  
    """
    y_int = [int(i) for i in y_color]
    return y_int


def noise_power(color_plot):
    """
    This function reads in a tdms file and returns a single value representing the average
    baseline noise for the entire colorplot.
    """
    ivsts = [color_plot[i] for i in range(1000)]
    
    for ivst in ivsts:
        maxes_idx = argrelmax(ivst)[0]
        mins_idx = argrelmin(ivst)[0]

        maxes = [ivst[i] for i in maxes_idx]
        mins = [ivst[i] for i in mins_idx]

    if len(maxes) > len(mins):
        del maxes[-1]
    if len(maxes) < len(mins):
        del mins[-1]

    return np.average(np.array(maxes)-np.array(mins))


def get_peak_heights(color_plot, x_color, y_color):
    """
    This function reads in a tdms file and returns a list of the peak heights in the 
    same order as the x values from preprocessing.read_tdms.  
    """    
    collection_rate=1

    x_int = [int(i) for i in x_color]
    y_int = [int(i) for i in y_color]

    # List of IvsTs with a peak
    ivsts = [color_plot[i] for i in y_int]

    # Take derivatives of IvsTs
    peak_and_times = []
    for idx,i in enumerate(y_int):
        ivst = color_plot[i]
        dy_1 = diff(ivst)/collection_rate
        dy_2 = diff(dy_1)/collection_rate

        peak = (x_int[idx], argrelmax(dy_2)[0])
        peak_and_times.append(peak)

    peak_with_time = []
    for i in peak_and_times:
        greater = []
        smaller = []
        equal = []
        for idx,time in enumerate(i[1]):
            if time < i[0]:
                smaller.append(time)
            #if time == i[0]:
            equal.append(time)
            if time > i[0]:
                greater.append(time)
        if len(smaller) != 0:
            start_time = smaller[-1]
        if len(smaller) == 0:
            start_time = equal[0]
        peak_with_time.append((i[0], start_time))

    heights = []  
    for idx,i in enumerate(peak_with_time):
        ivst = ivsts[idx]
        h = ivst[i[0]] - ivst[i[1]]
        heights.append(h)
        
    heights = [i/10 for i in heights]
    return heights


def get_signal_to_noise(color_plot, x_color, y_color):
    """
    This function chains noise_power and get_peak_heights together in order to compute the respective
    signal to noise ratios for the peaks returned by get_peak_heights.
    
    The returned ratios are in the same order as the x values from preprocessing.read_tdms.  
    """
    baseline_noise = noise_power(color_plot)
    peaks = get_peak_heights(color_plot, x_color, y_color)
    
    return [i/baseline_noise for i in peaks]


def get_decay_times(color_plot, x_color, y_color, collection_rate=1):
    """
    This function reads in a tdms file and returns a list of decay times for the peaks in the 
    same order as the x values from preprocessing.read_tdms.
    """

    x_int = [int(i) for i in x_color]
    y_int = [int(i) for i in y_color]
    
    # Take derivatives of IvsTs
    peak_and_times = []
    for idx,i in enumerate(y_int):
        ivst = color_plot[i]
        dy_1 = diff(ivst)/collection_rate
        dy_2 = diff(dy_1)/collection_rate

        peak = (x_int[idx], argrelmax(dy_2)[0])
        peak_and_times.append(peak)
        
    peak_with_time = []
    for i in peak_and_times:
        greater = []
        smaller = []
        equal = []
        for idx,time in enumerate(i[1]):
            if time < i[0]:
                smaller.append(time)
            equal.append(time)
            if time > i[0]:
                greater.append(time)
        if len(greater) != 0:
            end_time = greater[0]
        if len(greater) == 0:
            end_time = equal[0]
        peak_with_time.append((i[0], end_time))
    
    decay_times = [i[1]-i[0] for i in peak_with_time]
    decay_times = [i/10 for i in decay_times]
    
    decay_times = list(decay_times)
    for idx,i in enumerate(decay_times):
        #print(i)
        if abs(i) > 30:
            decay_times[idx] = 60-abs(i)
    
    return np.absolute(decay_times)


def get_start_and_end(color_plot, x_color, y_color):
    """
    This function reads in a tdms file and returns a list of tuples.
    The first value in the tuple is the start time for that peak, and the 
    second value is the end time for the peak.
    
    The list is in the same order as the x values from preprocessing.read_tdms.
    """
    x_int = [int(i) for i in x_color]
    y_int = [int(i) for i in y_color]
    collection_rate = 1
    
    # Take derivatives of IvsTs
    peak_and_times = []
    for idx,i in enumerate(y_int):
        ivst = color_plot[i]
        dy_1 = diff(ivst)/collection_rate
        dy_2 = diff(dy_1)/collection_rate
        peak = (x_int[idx], argrelmax(dy_2)[0])
        peak_and_times.append(peak)
        
    start_times = []
    for i in peak_and_times:
        greater = []
        smaller = []
        equal = []
        for idx,time in enumerate(i[1]):
            if time < i[0]:
                smaller.append(time)
            #if time == i[0]:
            equal.append(time)
            if time > i[0]:
                greater.append(time)
        if len(smaller) != 0:
            start_time = smaller[-1]
        if len(smaller) == 0:
            start_time = equal[0]
        start_times.append(start_time)

    end_times = [] 
    for i in peak_and_times:
        greater = []
        smaller = []
        equal = []
        for idx,time in enumerate(i[1]):
            if time < i[0]:
                smaller.append(time)
            equal.append(time)
            if time > i[0]:
                greater.append(time)
        if len(greater) != 0:
            end_time = greater[0]
        if len(greater) == 0:
            end_time = equal[0]
        end_times.append(end_time)
        
    return list(zip(start_times,end_times))


def isolate_peaks(color_plot, x_color, y_color, start_end):
    """
    This function reads in a tdms file and returns a list of arrays of variable lengths.
    
    The returned arrays represent the section of IvsT that contains the peak.
    
    The order is the same order as the x values from preprocessing.read_tdms.
    """


    # Get list of IvsTs
    y_int = [int(i) for i in y_color]

    ivsts = [color_plot[i] for i in y_int]

    short_arrays = []
    for idx,i in enumerate(start_end):
        ivst = ivsts[idx]
        start = i[0]-1
        end = i[1]+1
        short_arrays.append(ivst[start:end])

    return short_arrays


def normalize_ivsts(list_of_arrays):
    """
    This function takes the output of isolate peaks and returns a list of arrays that
    are normalized to the first value in the array.
    """
    normalized_list = []
    for i in list_of_arrays:
        normalized_array = []
        for j in i:
            difference = j - i[0]
            normalized_array.append(difference)
        normalized_list.append(normalized_array)
    
    return normalized_list


def get_auc(color_plot, x_color, y_color, start_end):
    """
    This function chains isolate_peaks and normalize_ivsts and calculates an integral
    of each normalized array.
    
    Returns a list of integrals in the same order as the x values from preprocessing.read_tdms.
    """
    
    peaks = isolate_peaks(color_plot, x_color, y_color, start_end)
    norm_peaks = normalize_ivsts(peaks)
    
    aucs = [trapz(i, dx=0.1) for i in norm_peaks]
    
    return np.absolute(aucs)