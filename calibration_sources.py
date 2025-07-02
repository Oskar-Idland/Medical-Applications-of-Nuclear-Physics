import pandas as pd

# Calibration sources dictionary
calibration_sources = {
    "Cs137": {"file": "../spectra/calibration/AA110625_Cs137.Spe", "name": "137CS", "Δt_c": 110},
    "Ba133": {"file": "../spectra/calibration/AB110625_Ba133.Spe", "name": "133BA", "Δt_c": 228},
    "Eu152": {"file": "../spectra/calibration/AC110625_Eu152.Spe", "name": "152EU", "Δt_c": 487},
}


# Activity and reference date info in a DataFrame
sources = pd.DataFrame([
    {"isotope": "137CS", "A0": 11.46e-6 * 3.7e10, "ref_date": "02/01/1978 12:00:00"},
    {"isotope": "133BA", "A0": 10.78e-6 * 3.7e10, "ref_date": "10/01/1988 12:00:00"},
    {"isotope": "152EU", "A0": 150000, "ref_date": "01/01/2002 12:00:00"},
])


"""
calib_path_Cs = spec_calib_path / 'AA110625_Cs137.Spe'
calib_path_Ba = spec_calib_path / 'AB110625_Ba133.Spe'
calib_path_Eu = spec_calib_path / 'AC110625_Eu152.Spe'

# Extract the spectrums for calibration
cb = ci.Calibration()
sp_Cs137 = ci.Spectrum(calib_path_Cs)
sp_Ba133 = ci.Spectrum(calib_path_Ba)
sp_Eu152 = ci.Spectrum(calib_path_Eu)

# Assign isotopes to the spectrums. Our calibration samples are 100% Cs137, Ba133, and Eu152
# Note: Isotope names must be uppercase
sp_Cs137.isotopes = ['137CS']
sp_Ba133.isotopes = ['133BA']
sp_Eu152.isotopes = ['152EU']


def Ci_to_Bq(Ci):
    """Convert Curie to Becquerel."""
    return Ci * 3.7e10  # 1 Ci = 3.7e10 Bq

# Add information about the sources
# Note: Isotope names must be uppercase
sources = [
    {'isotope': '133BA',
     'A0': Ci_to_Bq(10.78*1e-6),
     'ref_date': '10/01/1988 12:00:00'},
    {'isotope': '137CS', 
     'A0': Ci_to_Bq(11.46*1e-6), 
     'ref_date': '02/01/1979 12:00:00'},
    {'isotope': '152EU',
     'A0': 150*1e3,
     'ref_date': '01/01/2002 12:00:00'}
]

sources = pd.DataFrame(sources)
cb.calibrate([sp_Ba133, sp_Cs137, sp_Eu152], sources=sources)
cb.saveas(root_path / 'calibration.json')
"""

"""
import curie as ci
import pandas as pd
from pathlib import Path

class Path(Path):
    '''Wrapper for pathlib.Path to ensure string representation is used in Spectrum class.'''
    def endswith(self, suffix):
        return str(self).endswith(suffix)
    
    def split(self, sep=None):
        return str(self).split(sep)
        
   

root_test = Path(__file__).parent 

spec1 = root_test / 'AD022619_Cs137_31.6cm_HPGE.Spe'
spec2 = root_test / 'AE022619_Ba133_31.6cm_HPGE.Spe'
spec3 = root_test / 'AF022619_Eu152_31.6cm_HPGE.Spe'

cb = ci.Calibration()
sp_Cs137 = ci.Spectrum(spec1)
sp_Ba133 = ci.Spectrum(spec2)
sp_Eu152 = ci.Spectrum(spec3)

sp_Eu152.isotopes = ['152EU']
sp_Ba133.isotopes = ['133BA']
sp_Cs137.isotopes = ['137CS']

sources = [
        {'isotope':'133BA', 'A0':3.989E4, 'ref_date':'01/01/2009 12:00:00'},
        {'isotope':'137CS', 'A0':3.855E4, 'ref_date':'01/01/2009 12:00:00'},
        {'isotope':'152EU', 'A0':3.3929E4, 'ref_date':'01/01/2009 12:00:00'}
    ]

sources = pd.DataFrame(sources)

cb.calibrate([sp_Ba133, sp_Cs137, sp_Eu152], sources=sources)
# cb.saveas('calibration.json')
# cb.plot()






# print(sp_Eu152.summarize())
# sp_Eu152.plot()
# cb.plot()
"""