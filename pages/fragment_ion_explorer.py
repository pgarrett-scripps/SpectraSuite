import numpy as np
import streamlit as st
from spectra import read_ms2_file
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import pandas as pd
import plotly.express as px
from scipy import stats

from util import get_lines_from_uploaded_file

st.title('Fragment Ion Explorer')

ms2_file = st.file_uploader("Ms2 File", type=['ms2'])

BIN_RESOLUTION = 10_000
specta_peak_count_filter = st.number_input("Number of peaks to use per spectra", min_value=0, max_value=1000, value=20,
                        help='Max number of peaks to look at per spectra (sorted by highest -> lowest intensity)')
analysis_fragment_ion_count = st.number_input("Number of fragments to analyze", min_value=0, max_value=1000, value=20,
                        help='The number of fragment ions to plot/view (sorted by highest-> lowest frequency')
min_fragment_mz = st.number_input("Min fragment ion m/z", min_value=0, max_value=10_000, value=100,
                                  help='min fragment mz (ions less than this are excluded from analysis')
max_fragment_mz = st.number_input("Max fragment ion m/z", min_value=0, max_value=10_000, value=500,
                                  help='max fragment mz (ions greater than this are excluded from anlysis)')
fragment_ppm_tolerance = st.number_input('fragment ppm', min_value=0, max_value=10_000, value=50)
use_loss_fragments = st.checkbox("Use loss fragments", value=False,
                                 help='visualize loss fragments. All fragment ions are subtracted from the precursor ion mz value')
if ms2_file and st.button("Run"):
    ms2_lines = get_lines_from_uploaded_file(ms2_file)
    mzs, ints, charges, masses, specs = read_ms2_file(ms2_lines, min_fragment_mz, max_fragment_mz,
                                                      specta_peak_count_filter,
                                                      loss=use_loss_fragments)
    mzs, ints, specs = np.array(mzs, dtype='float32'), np.array(ints, dtype='float32'), np.array(
        specs, dtype='int32')
    num_spectra = len(set(specs))

    sort_indexes = np.argsort(mzs)
    mzs = mzs[sort_indexes]
    ints = ints[sort_indexes]
    specs = specs[sort_indexes]

    nbins = int(max_fragment_mz - min_fragment_mz)
    frequency_hist, frequency_bin_edges = np.histogram(mzs, bins=nbins)
    weighted_frequency_hist, weighted_frequency_bin_edges = np.histogram(mzs, weights=ints, bins=nbins)

    with st.expander("Histogram"):
        bar_df = pd.DataFrame(data={'edge': frequency_bin_edges[:-1],
                                    'frequency': frequency_hist,
                                    'weighted_frequency': weighted_frequency_hist})
        fig = px.bar(bar_df,
                     x='edge',
                     y=['frequency', 'weighted_frequency'],
                     barmode='group')
        st.plotly_chart(fig)

    sort_idx = np.argsort(weighted_frequency_hist)[::-1]

    data = {'peak': [], 'logprob': []}
    for idx in sort_idx[:analysis_fragment_ion_count]:
        lower_bin_edge = frequency_bin_edges[idx - 1]
        upper_bin_edge = frequency_bin_edges[min(idx + 2, len(frequency_bin_edges) - 1)]

        mz_mask = (upper_bin_edge >= mzs) & (mzs >= lower_bin_edge)
        mz_list = mzs[mz_mask]
        int_list = ints[mz_mask]

        if len(mz_list) == 0:
            continue

        # fit kde to -1 -> +1 bins (extract multiple peaks per bin)
        kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
        kde.fit(mz_list.reshape(-1, 1), frequency_bin_edges[idx + 1], 1000)
        X = np.linspace(frequency_bin_edges[idx], frequency_bin_edges[idx + 1], 500)
        logprob = np.exp(kde.score_samples(X.reshape(-1, 1)))
        peaks, _ = find_peaks(logprob, height=0)
        for peak in peaks:
            data['peak'].append(X[peak])
            data['logprob'].append(logprob[peak])

    df = pd.DataFrame(data=data)

    # Compute fragment ion stats
    num_samples, frequency, peak_mz_std, peak_mz_median, peak_mz_mean, peak_mz_sem, peak_int_std, peak_int_median, \
    peak_int_mean, peak_int_sem = [], [], [], [], [], [], [], [], [], []
    for _, row in df.iterrows():
        mz = row['peak']
        mz_offset = mz * fragment_ppm_tolerance / 1_000_000
        min_mz, max_mz = mz - mz_offset, mz + mz_offset
        mz_mask = (max_mz >= mzs) & (mzs >= min_mz)
        mz_list = mzs[mz_mask]
        int_list = ints[mz_mask]
        spec_list = specs[mz_mask]

        spec_dict = {}
        for mz, i, spec in zip(mz_list, int_list, spec_list):
            if spec not in spec_dict:
                spec_dict[spec] = {'mz': mz, 'i': i}
            elif i > spec_dict[spec]['i']:
                spec_dict[spec]['mz'] = mz
                spec_dict[spec]['i'] = i

        mz_list = [spec_dict[spec]['mz'] for spec in spec_dict]
        int_list = [spec_dict[spec]['i'] for spec in spec_dict]

        num_samples.append(len(mz_list))
        frequency.append(len(mz_list) / num_spectra)
        peak_mz_std.append(np.std(mz_list))
        peak_mz_median.append(np.median(mz_list))
        peak_mz_mean.append(np.mean(mz_list))
        peak_mz_sem.append(stats.sem(mz_list))
        peak_int_std.append(np.std(int_list))
        peak_int_median.append(np.median(int_list))
        peak_int_mean.append(np.mean(int_list))
        peak_int_sem.append(stats.sem(int_list))

    df['N'] = num_samples
    df['frequency'] = frequency
    df['mz_std'] = peak_mz_std
    df['mz_median'] = peak_mz_median
    df['mz_mean'] = peak_mz_mean
    df['mz_sem'] = peak_mz_sem

    df['int_std'] = peak_int_std
    df['int_median'] = peak_int_median
    df['int_mean'] = peak_int_mean
    df['int_sem'] = peak_int_sem

    df = df.sort_values(by='frequency', ascending=False)[:analysis_fragment_ion_count]
    df = df.sort_values(by='peak')

    with st.expander('Data'):
        st.dataframe(df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Frequency',
        x=df["peak"], y=df["frequency"]
    ))
    fig.add_trace(go.Bar(
        name='Intensity',
        x=df["peak"], y=df["int_mean"],
        error_y=dict(type='data', array=df['int_sem'])
    ))
    fig.update_layout(barmode='group')
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

    st.subheader("Fragment Ions")
    st.text(','.join(map(str, df['peak'])))
