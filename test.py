import curie as ci

spec1 = 'AD022619_Cs137_31.6cm_HPGE.Spe'
spec2 = 'AE022619_Ba133_31.6cm_HPGE.Spe'
spec3 = 'AF022619_Eu152_31.6cm_HPGE.Spe'
spec4 = 'AB02262019_Ni04_31.6cm_HPGE.Spe'


cb = ci.Calibration('calibration.json')
sp = ci.Spectrum(spec4)
sp.cb = cb
sp.isotopes = ['64CU', '62CU', '60CU', '57NI', '62CO', '61CO', '60CO', '59CO', '58CO', '57CO', '56CO', '56CO', '62FE', '61FE', '60FE', '59FE', '58FE', '57FE', '56FE', '55FE', '54FE', '53FE', '52FE', '56MN', '55MN', '54MN', '53MN', '52MN', '51MN', '51CR', '49CR', '48CR', '49V', '48V']
sp.fit_config = {'SNR_min': 3.5, 'dE_511': 9}
# sp.plot()
sp.saveas('peak_data_AB02262019_Ni04_31.6cm_HPGE.csv')
sp.saveas('peak_data_AB02262019_Ni04_31.6cm_HPGE.pdf')
sp.saveas('peak_data_AB02262019_Ni04_31.6cm_HPGE.json')