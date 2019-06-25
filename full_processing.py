# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:27:15 2019

@author: dmhedges
"""

def get_dataframe(filename):
    """
    This function links all the feature generator functions and exports all the results as a pandas dataframe.
    """
    import numpy as np
    import pandas as pd
    from numpy import ndarray
    from numpy import diff
    from numpy import trapz
    
    import nptdms
    from nptdms import TdmsFile
    
    import scipy
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters
    from scipy import stats
    from scipy import signal
    from scipy.signal import bessel
    from scipy.ndimage.filters import maximum_filter
    from scipy.stats import moment
    from scipy.signal import argrelmin
    from scipy.signal import argrelmax
    
    from peakutils.peak import indexes
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    #% matplotlib inline
    
    from volt_analysis_funcs import read_tdms
    from volt_analysis_funcs import get_peak_indices
    from volt_analysis_funcs import plot_heat_peaks
    from volt_analysis_funcs import filter_by_ox_index
    from volt_analysis_funcs import preprocessing
    from volt_analysis_funcs import get_cvs
    from volt_analysis_funcs import get_r2
    from volt_analysis_funcs import get_rise_times
    from volt_analysis_funcs import get_oxid_idx
    from volt_analysis_funcs import noise_power
    from volt_analysis_funcs import get_peak_heights
    from volt_analysis_funcs import get_signal_to_noise
    from volt_analysis_funcs import get_decay_times
    from volt_analysis_funcs import get_start_and_end
    from volt_analysis_funcs import isolate_peaks
    from volt_analysis_funcs import normalize_ivsts
    from volt_analysis_funcs import get_auc
    
    from peak_finder_2 import find_peaks
    
    color_plot = read_tdms(filename)
    #x,y = get_peak_indices(color_plot)

    x_color,y_color = find_peaks(filename)

    start_end = get_start_and_end(color_plot, x_color, y_color)

    cvs = get_cvs(color_plot, x_color,y_color)
    df = pd.read_csv('sample_volt.txt', header=None, delimiter='\t')
    volt = list(df[1])
    r2 = get_r2(cvs, volt)
    rise_times = get_rise_times(color_plot, x_color, y_color)
    oxid_idx = get_oxid_idx(x_color, y_color)
    snr = get_signal_to_noise(color_plot, x_color, y_color)
    decay_times = get_decay_times(color_plot, x_color, y_color, collection_rate=1)
    get_start_and_end(color_plot, x_color, y_color)
    isolate_peaks(color_plot, x_color, y_color, start_end)
    aucs = get_auc(color_plot, x_color, y_color, start_end)
    peak_heights = get_peak_heights(color_plot, x_color, y_color)
    
    
    df = pd.DataFrame(
    {'Rise Time': rise_times,
     'Decay Time': decay_times,
     'Oxidation Index': oxid_idx,
     'R Squared Voltammogram': r2,
     'Signal to Noise Ratio': snr,
     'Peak Height': peak_heights,
     'Area Under the Curve': aucs})
    
    return df