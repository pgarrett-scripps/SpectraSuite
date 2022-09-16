from pathlib import Path

import streamlit as st
from senpy.dtaSelectFilter.parser2 import read_file

from util import get_lines_from_uploaded_file

st.title("Filter MS2")

ms2_file = st.file_uploader("Ms2 File", type=['ms2'])
dta_filter_file = st.file_uploader(label="DTASelect-filter.txt", type=".txt")

with st.expander("Help"):
    st.markdown("""
    Ensure that the ms2 file matches the notation used in the DTASelect-filter.txt 
    
    The app matches the ms2 file name (without extension) to the file name in the DTASelect-filter file.
    
    (ms2 file name) 169.ms2 -> 169
    
    (ms2 file name) 169a.ms2 -> 169a

    
    (DTASelect-filter.txt line) * ***169***.36947.36947.2 [...]   --> 169
    
    (DTASelect-filter.txt line) * ***169a***.36947.36947.2 [...]   --> 169a
    
    """)

if st.button("Run"):
    dta_filter_file_name = Path(dta_filter_file.name).stem
    ms2_file_name = Path(ms2_file.name).stem

    dta_filter_lines = get_lines_from_uploaded_file(dta_filter_file)
    _, dta_filter_results, _ = read_file(dta_filter_lines)

    ms2_lines = get_lines_from_uploaded_file(ms2_file)

    scan_numbers = set()
    for dta_filter_result in dta_filter_results:

        protein_locuses = [protein.locus for protein in dta_filter_result.proteins]
        if all(['Reverse_' in locus or 'contaminant_' in locus for locus in protein_locuses]):
            continue
        for peptide in dta_filter_result.peptides:
            if peptide.file_name == ms2_file_name:
                scan_numbers.add(int(peptide.low_scan))

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
    st.download_button(label="Download Filter MS2", data="\n".join(ms2_lines) + "\n", file_name=f"{ms2_file_name}_{dta_filter_file_name}.ms2")






