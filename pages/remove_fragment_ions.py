import streamlit as st

from util import get_lines_from_uploaded_file

st.title("Remove Fragment Ions")


ms2_file = st.file_uploader("Ms2 File", type=['ms2'])
fragments = st.text_input("Fragment Ions (comma separated)")
ppm = st.number_input("Fragment Ion match PPM", 50)

if st.button("Run"):
    fragments = list(map(float, fragments.strip().split(",")))
    max_frag = max(fragments)
    ms2_lines = get_lines_from_uploaded_file(ms2_file)

    fragment_indexes = []
    for i, line in enumerate(ms2_lines):
        if line and line[0].isnumeric():
            mz = float(line.split(" ")[0])

            if mz > max_frag + 1:
                continue

            for frag_ion in fragments:
                if abs(mz - frag_ion)/mz*1_000_000 <= ppm:
                    fragment_indexes.append(i)
                    break

    for i in sorted(fragment_indexes, reverse=True):
        del ms2_lines[i]

    st.download_button(label="Download Filter MS2", data="\n".join(ms2_lines) + "\n", file_name=ms2_file.name + "_no_frag.ms2")
