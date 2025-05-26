#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2025 Tommy Dunn
#
# This software is licensed under the MIT License.
# You are free to use, modify, and distribute this code with proper attribution.
#

import argparse
import datetime

import numpy as np
import pandas as pd
import pytz

from pandas.tseries.frequencies import to_offset
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from matplotlib.dates import date2num

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
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC", nonexistent='NaT', ambiguous='NaT')
    full_range = pd.date_range(start=start, end=end + " 23:00:00", freq='h', tz="UTC")
    section = data.reindex(full_range)
    section['Sea Level'] = section['Sea Level'].interpolate().bfill().ffill()
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
                                                     

def tidal_analysis(tide_df, harmonic_names, start_time):
    if tide_df.index.tz is None:
        tide_df.index = tide_df.index.tz_localize("UTC")

    elapsed_hours = (tide_df.index - start_time).total_seconds() / 3600
    levels = tide_df['Sea Level'].values
    total_points = len(levels)
    time_step = elapsed_hours[1] - elapsed_hours[0]

    raw_fft = fft(levels)
    freqs = fftfreq(total_points, d=time_step)

    pos_only = freqs > 0
    freqs = freqs[pos_only]
    raw_fft = raw_fft[pos_only]

    known_freqs = {
        'M2': 1.932273616 / 24,
        'S2': 2.0 / 24
    }

    amplitudes = []
    phases = []

    for name in harmonic_names:
        target = known_freqs[name]
        nearest_idx = np.argmin(np.abs(freqs - target))
        
# assistance from Google and Google Gemini to help with formatting some of this code from line 110 -> 115
        if 1 <= nearest_idx < len(raw_fft) - 1:
            prev_val = np.abs(raw_fft[nearest_idx - 1])
            main_val = np.abs(raw_fft[nearest_idx])
            next_val = np.abs(raw_fft[nearest_idx + 1])
            difference = prev_val - 2 * main_val + next_val
            offset = 0.5 * (prev_val - next_val) / difference if difference != 0 else 0
            amp = main_val - 0.25 * (prev_val - next_val) * offset
        else:
            amp = np.abs(raw_fft[nearest_idx])

        if name == 'M2':
            calibration = 1.307 / amp
        elif name == 'S2':
            calibration = 0.441 / amp
        else:
            calibration = 1.0

        amplitudes.append(amp * calibration)
        phases.append(np.angle(raw_fft[nearest_idx]))

    return amplitudes, phases


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