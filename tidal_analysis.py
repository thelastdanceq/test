#!/usr/bin/env python3
"""
Tidal Analysis Module

This module provides functions for analyzing tidal data,
including reading tidal data from files or directories,
performing tidal analysis, extracting specific data segments,
and calculating sea level rise.

Usage:
    $ python tidal_analysis.py directory_path

"""
import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import uptide


def read_tidal_data(filename_or_directory):
    """
    Read tidal data from file or directory.

    Args:
        filename_or_directory (str): Path to file or directory.

    Returns:
        pd.DataFrame: Tidal data.
    """
    data = []

    if os.path.isdir(filename_or_directory):
        # If directory is provided, read all files within the directory
        directory = filename_or_directory
        filenames = [os.path.join(directory, f)
                     for f in os.listdir(directory) if f.endswith(".txt")]
    else:
        # If a single file is provided, use that file
        filenames = [filename_or_directory]

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Skip headers
            data_lines = lines[11:]
            # Parse data
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 5:
                    date_time = parts[1] + ' ' + parts[2]

                    # parsing sea level
                    sea_level_str = parts[3]
                    if sea_level_str.endswith(('M', 'T', 'N')):
                        sea_level = np.nan
                    else:
                        sea_level = float(sea_level_str)

                    data.append((date_time,  sea_level))

    df = pd.DataFrame(data, columns=['Time', 'Sea Level'])
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
    df.set_index('Time', inplace=True, drop=False)

    return df


def extract_single_year_remove_mean(year, data):
    """
    Extract data for a single year and remove mean sea level.

    Args:
        year (str): Year to extract.
        data (pd.DataFrame): Tidal data.

    Returns:
        pd.DataFrame: Year data with mean sea level removed.
    """
    year_data = data[data.index.year == int(year)]

    mean_sea_level = year_data['Sea Level'].mean()

    year_data['Sea Level'] -= mean_sea_level

    return year_data


def extract_section_remove_mean(start, end, data):
    """
    Extract data for a section and remove mean sea level.

    Args:
        start (str): Start date.
        end (str): End date.
        data (pd.DataFrame): Tidal data.

    Returns:
        pd.DataFrame: Section data with mean sea level removed.
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end, format='%Y%m%d') + \
        pd.DateOffset(days=1) - pd.DateOffset(seconds=1)

    section_data = data[(data.index >= start_date) & (data.index <= end_date)]

    mean_sea_level = section_data['Sea Level'].mean()

    section_data['Sea Level'] -= mean_sea_level

    return section_data


def join_data(data1, data2):
    """
    Join two datasets.

    Args:
        data1 (pd.DataFrame): First dataset.
        data2 (pd.DataFrame): Second dataset.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    combined_df = pd.concat([data1, data2])

    combined_df.sort_index(inplace=True)

    return combined_df


def sea_level_rise(data):
    """
    Calculate sea level rise.

    Args:
        data (pd.DataFrame): Tidal data.

    Returns:
        float: Sea level rise slope.
        float: p-value.
    """
    clean_data = data.dropna(subset=['Sea Level'])
    start_date = clean_data.index.min()
    clean_data['Time_Fractional'] = data.index.to_series().apply(
        lambda x: (x - start_date).total_seconds() / (3600 * 24))

    result = linregress(
        clean_data['Time_Fractional'], clean_data['Sea Level'])

    return result.slope, result.pvalue


def tidal_analysis(data, constituents, start_datetime):
    """
    Perform tidal analysis on the provided data.

    Args:
        data (pd.DataFrame): Tidal data.
        constituents (list): List of tidal constituents to analyze.
        start_datetime (pd.Timestamp): Start datetime for analysis.

    Returns:
        tuple: Amplitude and phase of tidal constituents.
    """
    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_datetime)

    times = []
    eta = []
    for index, row in data.iterrows():
        if not pd.isna(row['Sea Level']):
            times.append((index.tz_localize(start_datetime.tzinfo) -
                         start_datetime).total_seconds())
            eta.append(row['Sea Level'])

    amp, pha = uptide.harmonic_analysis(tide, eta, times)

    return amp, pha


def get_longest_contiguous_data(data):
    """
    Identify the longest contiguous segment of data based only on 'Sea Level'.

    Args:
        data (pd.DataFrame): Tidal data with 'Time' and 'Sea Level' columns.

    Returns:
        pd.DataFrame: The longest contiguous segment of tidal data.
    """
    valid_data_mask = data['Sea Level'].notna()


    # Compute the difference from the previous row to detect changes (0 -> 1 or 1 -> 0)
    # Replace NaN in the first element (result of diff) with 0 to avoid interpretation errors
    # Compare each element to 0 to determine if there is a change (True if changed, False if not)
    # Cumulatively sum the True values to assign a unique block identifier to each segment
    blocks = valid_data_mask.astype(int)\
        .diff()\
        .fillna(0)\
        .ne(0)\
        .cumsum()\

    filtered_data = data[valid_data_mask]

    longest_block = filtered_data.groupby(
        blocks[valid_data_mask]).size().idxmax()

    return filtered_data[blocks[valid_data_mask] == longest_block]



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
    dataframe = read_tidal_data(dirname)
    amplitude, phase = tidal_analysis(
        dataframe, ['M2', 'S2'], dataframe.index.min())
    print("Tidal constituents for ${}:", dirname)
    print("Amplitude:", amplitude)
    print("Phase:", phase)
