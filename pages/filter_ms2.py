import streamlit as st
from senpy.dtaSelectFilter.parser2 import read_file

from util import get_lines_from_uploaded_file

ms2_file = st.file_uploader("Ms2 File", type=['ms2'])
dta_filter_file = st.file_uploader(label="DTASelect-filter.txt", type=".txt")

if st.button("Run"):
    dta_filter_lines = get_lines_from_uploaded_file(dta_filter_file)
    _, dta_filter_results, _ = read_file(dta_filter_lines)

    scan_numbers = set()
    for dta_filter_result in dta_filter_results:
        protein_locuses = [protein.locus for protein in dta_filter_result.proteins]
        if all(['Reverse_' in locus or 'contaminant_' in locus for locus in protein_locuses]):
            continue
        for peptide in dta_filter_result.peptides:
            scan_numbers.add(int(peptide.low_scan))

    ms2_lines = get_lines_from_uploaded_file(ms2_file)

    flags = []
    skip_lines = False
    for line in ms2_lines:
        if line.startswith('S'):
            skip_lines = False
            scan_num = int(line.rstrip().split('\t')[1])
            if scan_num not in scan_numbers:
                skip_lines = True

        if skip_lines:
            flags.append(False)
        else:
            flags.append(True)

    ms2_lines = [line for i, line in enumerate(ms2_lines) if flags[i]]
    st.download_button(label="Download Filter MS2", data="\n".join(ms2_lines) + "\n", file_name=ms2_file.name + "_filtered.ms2")






