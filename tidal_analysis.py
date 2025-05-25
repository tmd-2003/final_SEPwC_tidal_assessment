#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 17:26:16 2025

@author: tommydunn
"""

#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import pytz 
from scipy.stats import linregress
from matplotlib.dates import date2num
from scipy.fft import rfft, rfftfreq
import datetime
from scipy.fft import fft, fftfreq
from pandas.tseries.frequencies import to_offset




def read_tidal_data(filename): #test completed
    df = pd.read_csv(
        filename,
        skiprows=11,
        sep=r'\s+',
        header=None,
        engine='python',
        on_bad_lines='skip'
    )
    df = df[[1, 2, 3]]
    df.columns = ['Date', 'Time', 'Sea Level']
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df.replace(to_replace=r'.*[MNT]$', value={'Sea Level': np.nan}, regex=True, inplace=True)
    df['Sea Level'] = pd.to_numeric(df['Sea Level'], errors='coerce')
    df['Time'] = df['Time']  # (retained for downstream test compatibility)
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    return df[['Sea Level', 'Time']]


def join_data(data1, data2): #test completed
    combined = pd.concat([data1, data2])
    combined = combined.sort_index()
    return combined


def extract_section_remove_mean(start, end, data):
    full_range = pd.date_range(start=start, end=end + " 23:00:00", freq='h')
    section = data.reindex(full_range)
    section['Sea Level'] = section['Sea Level'].interpolate()
    section['Sea Level'] = section['Sea Level'].bfill().ffill()
    section['Sea Level'] -= section['Sea Level'].mean()
    return section



def extract_single_year_remove_mean(year, data): #test completed
    start = f"{year}-01-01 00:00:00"
    end = f"{year}-12-31 23:00:00"
    full_range = pd.date_range(start=start, end=end, freq='h')
    year_data = data.loc[start:end]
    year_data = year_data.reindex(full_range)
    year_data['Sea Level'] = year_data['Sea Level'].interpolate().bfill().ffill()
    year_data['Sea Level'] -= year_data['Sea Level'].mean()
    return year_data

def sea_level_rise(data):

    # Drop rows where Sea Level is NaN
    clean_data = data.dropna(subset=['Sea Level'])
    time_numeric = date2num(clean_data.index)

    # Perform linear regression
    slope_per_day, _, _, p_value, _ = linregress(time_numeric, clean_data['Sea Level'].values)
    # Convert slope from per-day to per-year
    slope_per_year = slope_per_day * 365.25

    return slope_per_year, p_value
                                                     



#TO DO NEXT/PROBLEM LOG: 
    # Ensure index is timezone aware (MATCH UTC) - as test fails due to time zone missmatch, RIP!
    # Convert datetime index to hours since start
    
def tidal_analysis(data, constituents, start_datetime):
  
    # Convert datetime index to hours since start
    time_hours = (data.index - start_datetime).total_seconds() / 3600
    sea_level = data['Sea Level'].values
    n = len(sea_level)
    dt = (time_hours[1] - time_hours[0])  # assumed uniform
    fft_vals = fft(sea_level)
    fft_freq = fftfreq(n, d=dt)

    # Only positive frequencies
    pos_mask = fft_freq > 0
    fft_freq = fft_freq[pos_mask]
    fft_vals = fft_vals[pos_mask]

    freq_map = {'M2': 1.932274, 'S2': 2.0}  # cycles per day
    amp_results = []
    phase_results = []

    for constituent in constituents:
        target_freq = freq_map.get(constituent)
        if target_freq is None:
            amp_results.append(np.nan)
            phase_results.append(np.nan)
            continue

        # convert to cycles per hour
        target_freq /= 24

        idx = np.argmin(np.abs(fft_freq - target_freq))
 
    # needed some help from Gemini Google AI from this section... to:
        # Quadratic interpolation
        if 1 <= idx < len(fft_vals) - 1:
            y0, y1, y2 = np.abs(fft_vals[idx - 1]), np.abs(fft_vals[idx]), np.abs(fft_vals[idx + 1])
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0
            refined_amp = y1 - 0.25 * (y0 - y2) * delta
        else:
            refined_amp = np.abs(fft_vals[idx])
    
    
        amp_results.append(refined_amp)
        phase_results.append(np.angle(fft_vals[idx]))
# this bit

    return amp_results, phase_results





def get_longest_contiguous_data(data):


    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2024, Jon Hill"
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose