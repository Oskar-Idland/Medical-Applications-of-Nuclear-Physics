#!/usr/bin/env python
# coding: utf-8

# ## Imports & Setup

# In[1]:


import curie as ci
import pandas as pd
import pathlib

class Path(pathlib.Path):
    '''Wrapper for pathlib.Path to ensure string representation is used in Spectrum class.'''
    def endswith(self, suffix):
        return str(self).endswith(suffix)
    
    def split(self, sep=None):
        return str(self).split(sep)
    
root_path = Path.cwd().parent 

spec_path  = root_path / 'spectra'
calib_path = spec_path / 'calibration'
exp_path   = spec_path / 'experiment'
test_path  = spec_path / 'test'


# ## Calibration

# In[2]:


calib_path_Cs = calib_path / 'AA110625_Cs137.Spe'
calib_path_Ba = calib_path / 'AB110625_Ba133.Spe'
calib_path_Eu = calib_path / 'AC110625_Eu152.Spe'

# Extract the spectrums for calibration
cb = ci.Calibration()
sp_Cs137 = ci.Spectrum(calib_path_Cs)
sp_Ba133 = ci.Spectrum(calib_path_Ba)
sp_Eu152 = ci.Spectrum(calib_path_Eu)

# Assign isotopes to the spectrums. Our calibration samples are 100% Cs137, Ba133, and Eu152
sp_Cs137.isotopes = ['137Cs']
sp_Ba133.isotopes = ['133Ba']
sp_Eu152.isotopes = ['152Eu']

# Add information about the sources
def Ci_to_Bq(Ci):
    """Convert Curie to Becquerel."""
    return Ci * 3.7e10  # 1 Ci = 3.7e10 Bq

# Note: Isotope names must be uppercase
sources = [
    {'isotope': '137CS', 
     'A0': Ci_to_Bq(11.46*1e-6), 
     'ref_date': '02/01/1979 12:00:00'},
    {'isotope': '133BA',
     'A0': Ci_to_Bq(10.78*1e-6),
     'ref_date': '10/01/1988 12:00:00'},
    {'isotope': '152EU',
     'A0': 150*1e3,
     'ref_date': '01/01/2002 12:00:00'}
]

sources = pd.DataFrame(sources)
cb.calibrate([sp_Ba133, sp_Cs137, sp_Eu152], sources=sources)
cb.plot()

