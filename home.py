import streamlit as st

st.title("Spectra Suite")

st.markdown("""
**What does each tab do?**

1) filter ms2: Filters a ms2 file to contain only spectra identified in the uploaded DTASelect-filter.txt file

2) fragment ion explorer: Looks for common fragment ions in all spectra in an uploaded ms2 file

3) remove fragment ions: Removes a list of fragment ions from an ms2 file within a given mass offset


**Why Remove Fragments?**

Search engines rely on comparing theoretical to experimental mass spectra. 
Theoretical spectra are derived from in silico fragmented peptides, which only contain 
ions from the peptide. Experimental spectra are often missing fragment ions, or contain extra 
ions not found in the theoretical spectra. Extra Ions (especially high intensity ions) lower
the spectral similarity scores and can reduce the number of peptide/protein IDs. For example ions which 
occur due to fragmented tags or ptms don't offer sequential information and are not included
in the theoretical spectra. Thus removing these 'extra' ions from the experimental will remove the effect 
and improve similarity scores, leading to an increase in IDs.

**Tips:**

I would recommend filtering large ms2 files with the "filter ms2" tab prior to analyzing the fragment ions. 
Doing so will greatly speed up the the analysis.

You can download ms2 files from ip2 by clicking on the spectra tab in your project. 
By default it will download as ms2_name.ms2.txt so make sure it is saved as just ms2_name.ms2 (without the .txt)

Also, since all DTASelect-filter.txt files share the same name it is good practice to rename it to match the project 
and/or search conditions. For example: ANL_project_XXXX_unmod_DTASelect-filter.txt or something similar.
""")
