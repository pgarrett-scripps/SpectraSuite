from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
import streamlit as st
import math
from spectra import read_ms2_file
import plotly.express as px
import plotly.express as px

from util import get_lines_from_uploaded_file

st.title('Fragment Ion Explorer')

with st.expander("Help"):
    st.markdown("""
    
        **Figure Bars:**
    
        **frequency** - is the frequency that a fragment ion is found in spectra
        (0.0 means it never occurs while 1.0 means it occurs in every spectra)

        **intensity** - the intensity of the fragment ion relative to the max peak in each spectra
        
        **Note:** it is possible to have a frquency and intensity > 1.0. This occurs when there are multile ions
        close to the fragment ion mass.
        """)

ms2_file = st.file_uploader("Ms2 File", type=['ms2'])

BIN_RESOLUTION = 10_000
SPECTRA_FILTER_N = st.number_input("Consider top N most intense peaks per spectra", min_value=0, max_value=1000, value=20)
PLOT_FILTER_N = st.number_input("Plot Top N peaks", min_value=0, max_value=1000, value=20)
MIN_MZ = st.number_input("Min MZ", min_value=0, max_value=10_000, value=100)
MAX_MZ = st.number_input("Max MZ", min_value=0, max_value=10_000, value=500)

if ms2_file and st.button("Run"):
    ms2_lines = get_lines_from_uploaded_file(ms2_file)
    all_mz, all_ints, all_charges, all_masses = read_ms2_file(ms2_lines, MIN_MZ, MAX_MZ, SPECTRA_FILTER_N)

    all_mz, all_ints = np.array(all_mz, dtype='float32'), np.array(all_ints, dtype='float32')

    num_spectra = len(all_charges)

    def get_bin(mz):
        return math.floor((mz - MIN_MZ) / (MAX_MZ - MIN_MZ) * BIN_RESOLUTION)

    def get_value_from_bin(bin):
        val_lower = bin * (MAX_MZ - MIN_MZ) / BIN_RESOLUTION + MIN_MZ
        val_upper = (bin + 1) * (MAX_MZ - MIN_MZ) / BIN_RESOLUTION + MIN_MZ
        return val_lower, val_upper

    def get_bins_from_spectras(spectras, use_intensity=False):
        bins = np.zeros(BIN_RESOLUTION, dtype='float32')
        for spectra in spectras:
            max_intensity = max(spectra.int_spectra)
            spectra.sort_by_intensity()
            for mz, i in zip(spectra.mz_spectra[-SPECTRA_FILTER_N:], spectra.int_spectra[-SPECTRA_FILTER_N:]):
                bin = get_bin(mz)
                if use_intensity:
                    bins[bin] += i/max_intensity
                else:
                    bins[bin] += 1
        return bins


    def get_bins_from_all_spectra(all_mz, all_ints, use_intensity=False):
        bins = np.zeros(BIN_RESOLUTION, dtype='float32')
        for mz, i in zip(all_mz, all_ints):
            bin = get_bin(mz)
            if use_intensity:
                bins[bin] += i
            else:
                bins[bin] += 1
        return bins

    bin = get_bin(1000)
    val_lower, val_upper = get_value_from_bin(bin)
    print(1000, bin, val_lower, val_upper, (val_upper - val_lower)/2)

    #bins = get_bins_from_all_spectra(all_mz, all_ints)
    cnt_bins = get_bins_from_all_spectra(all_mz, all_ints, use_intensity=False)
    int_bins = get_bins_from_all_spectra(all_mz, all_ints, use_intensity=True)

    def sort_bins(bins):
        bins, indexes = zip(*sorted(zip(bins, list(range(BIN_RESOLUTION))), reverse=True))
        return indexes

    cnt_bin_indexes = sort_bins(cnt_bins)[:PLOT_FILTER_N]
    int_bin_indexes = sort_bins(int_bins)[:PLOT_FILTER_N]

    all_indexes = list(set(cnt_bin_indexes + int_bin_indexes))

    top_n_cnt_bins = cnt_bins[all_indexes]
    top_n_int_bins = int_bins[all_indexes]

    df = pd.DataFrame(data={'bin_index': all_indexes, 'count': list(map(int,top_n_cnt_bins)), 'intensity':top_n_int_bins})
    df['frequency'] = df['count']/num_spectra
    df['intensity'] = df['intensity']/df['count']
    mz_bounds = list(map(get_value_from_bin, df['bin_index']))
    df['mz_lower'] = [mz_bound[0] for mz_bound in mz_bounds]
    df['mz_upper'] = [mz_bound[1] for mz_bound in mz_bounds]

    idx = np.argsort(all_mz)
    all_mz = all_mz[idx]
    all_ints = all_ints[idx]
    data = {'bin_index':[], 'mz':[], 'intensity':[]}
    stats_data = {'bin_index':[], 'mean_mz':[], 'median_mz': [], 'std_mz':[]}
    for _, row in df.iterrows():
        lower_index = bisect_right(all_mz, row['mz_lower'])
        upper_index = bisect_left(all_mz, row['mz_upper'])
        matching_mz = all_mz[lower_index:upper_index]
        matching_int = all_ints[lower_index:upper_index]
        median_mz = np.median(matching_mz)
        mean_mz = np.mean(matching_mz)
        std_mz = np.std(matching_mz)
        stats_data['bin_index'].append(row['bin_index'])
        stats_data['mean_mz'].append(mean_mz)
        stats_data['median_mz'].append(median_mz)
        stats_data['std_mz'].append(std_mz)

        for mz, i in zip(matching_mz, matching_int):
            data['bin_index'].append(row['bin_index'])
            data['mz'].append(mz)
            data['intensity'].append(i)

    tmp_df = pd.DataFrame(data=stats_data)
    df = pd.merge(df, tmp_df, on='bin_index')
    df = df.sort_values(by=['mean_mz'])
    st.dataframe(df)

    fig = px.bar(df, x=[round(x,4) for x in df["mean_mz"]], y=['frequency', 'intensity'],barmode='group')

    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fragment Ions")
    st.text(','.join(map(str, df['mean_mz'])))


