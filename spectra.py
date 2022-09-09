from dataclasses import dataclass, field
import numpy as np
from typing import List
import streamlit as st

@dataclass
class Spectra:
    mz_spectra: List = field(default_factory = lambda: [])
    int_spectra: List = field(default_factory = lambda: [])

    def sort_by_intensity(self):
        if self.int_spectra and self.mz_spectra:
            self.int_spectra, self.mz_spectra = zip(*sorted(zip(self.int_spectra , self.mz_spectra), reverse=True))

    def normalize_intensity(self):
        if self.int_spectra and self.mz_spectra:
            max_int = max(self.int_spectra)
            self.int_spectra = [i/max_int for i in self.int_spectra]

    def filter_mz(self, min_mz, max_mz):
        if self.int_spectra and self.mz_spectra:
            filtered_spectra = list(zip(*filter(lambda x: min_mz <= x[1] <= max_mz, zip(self.int_spectra, self.mz_spectra))))
            if filtered_spectra:
                self.int_spectra, self.mz_spectra = filtered_spectra
            else:
                self.int_spectra, self.mz_spectra = [], []


def read_ms2_file(ms2_lines: list[str], min_mz, max_mz, n):
    spectra = None
    all_mz, all_int, all_charges, all_masses = [], [], [], []

    for line in ms2_lines:
        if not line:
            continue
        if line.startswith("H"):
            continue
        elif line.startswith("S"):
            if spectra != None:
                spectra.filter_mz(min_mz, max_mz)
                spectra.normalize_intensity()
                spectra.sort_by_intensity()

                all_mz.extend(spectra.mz_spectra[:n])
                all_int.extend(spectra.int_spectra[:n])

            spectra = Spectra()
        elif line.startswith("I"):
            continue
        elif line.startswith("Z"):
            elems = line.rstrip().split('\t')
            all_charges.append(int(elems[1]))
            all_masses.append(float(elems[2]))
        else:
            elems = line.rstrip().split(' ')
            mz, intensity = np.float32(elems[0]), np.float32(elems[1])
            spectra.mz_spectra.append(mz)
            spectra.int_spectra.append(intensity)

    return all_mz, all_int, all_charges, all_masses

