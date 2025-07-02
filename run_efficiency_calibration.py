from efficiency_calibration import EfficiencyCalibration
from calibration_sources import calibration_sources, sources

if __name__ == "__main__":
    calibrator = EfficiencyCalibration(calibration_sources, sources)
    calibrator.load_spectra()
    calibrator.calibrate()
    calibrator.fit_peaks()
    calibrator.plot_and_save()
    print("Efficiency calibration complete and saved.")