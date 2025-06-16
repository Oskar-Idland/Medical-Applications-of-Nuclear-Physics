import curie as ci
import pandas as pd

spec1 = 'AD022619_Cs137_31.6cm_HPGE.Spe'
spec2 = 'AE022619_Ba133_31.6cm_HPGE.Spe'
spec3 = 'AF022619_Eu152_31.6cm_HPGE.Spe'

cb = ci.Calibration()
sp_Cs137 = ci.Spectrum(spec1)
sp_Ba133 = ci.Spectrum(spec2)
sp_Eu152 = ci.Spectrum(spec3)

sp_Eu152.isotopes = ['152EU']

print(sp_Eu152.isotopes)

sources = [
        {'isotope':'133BA', 'A0':3.989E4, 'ref_date':'01/01/2009 12:00:00'},
        {'isotope':'137CS', 'A0':3.855E4, 'ref_date':'01/01/2009 12:00:00'},
        {'isotope':'152EU', 'A0':3.3929E4, 'ref_date':'01/01/2009 12:00:00'}
    ]

sources = pd.DataFrame(sources)

cb.calibrate([sp_Ba133, sp_Cs137, sp_Eu152], sources=sources)
cb.saveas('calibration.json')







# print(sp_Eu152.summarize())
# sp_Eu152.plot()
# cb.plot()