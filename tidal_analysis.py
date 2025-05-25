#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np


def read_tidal_data(filename):
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
    df['Time'] = df['Time']  # retained for downstream test compatibility
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    return df[['Sea Level', 'Time']]


def join_data(data1, data2):
    combined = pd.concat([data1, data2])
    combined = combined.sort_index()
    return combined


def extract_section_remove_mean(start, end, data):
    """
    Extracts a time slice of the tidal data between 'start' and 'end' datetimes,
    reindexes to include missing hours, interpolates and zero-centers sea level.
    """
    full_range = pd.date_range(start=start, end=end, freq='h')
    section = data.loc[start:end]
    section = section.reindex(full_range)
    section['Sea Level'] = section['Sea Level'].interpolate()
    section['Sea Level'] = section['Sea Level'].bfill().ffill()
    section['Sea Level'] -= section['Sea Level'].mean()
    return section

def join_data(data1, data2):

    return 



def sea_level_rise(data):

                                                     
    return 

def tidal_analysis(data, constituents, start_datetime):


    return 

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