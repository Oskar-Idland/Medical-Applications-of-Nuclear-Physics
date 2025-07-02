import curie as ci
import pandas as pd

class EfficiencyCalibration:
    """Class to handle the efficiency calibration of gamma spectra using multiple sources."""
    def __init__(self, calibration_sources, source_info):
        """Initialize the calibration with sources and their information."""
        self.calibration_sources = calibration_sources
        self.source_info = source_info
        self.spectra = []
        self.calibration = ci.Calibration()
    
    def load_spectra(self):
        """Load spectra from the calibration sources."""
        self.spectra = []
        for data in self.calibration_sources.values():
            spe = ci.Spectrum(data["file"])
            spe.isotopes = [data["name"]]
            self.spectra.append(spe)
    
    def calibrate(self):
        """Calibrate the spectra using the provided calibration sources."""
        self.calibration.calibrate(self.spectra, self.source_info)
        
    def fit_peaks(self):
        """Fit peaks in the spectra using the gamma energies from the calibration sources."""
        for spe in self.spectra:
            spe.auto_calibrate()
            _ = spe.fit_peaks(bg="snip", SNR_min=3, multi_max=2, ident_idx=0)
    
    def plot_and_save(self, filename="detector_efficiency_calibration.json", plot_filename="efficiency_calibration_plot.pdf"):
        """Plot the efficiency calibration and save the results."""
        self.calibration.plot_effcal(saveas=plot_filename, show=False)
        self.calibration.saveas(filename)