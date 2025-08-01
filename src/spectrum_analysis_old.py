import curie as ci
import numpy as np
import pandas as pd
from path import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from uncertainties import ufloat, UFloat
from uncertainties.umath import exp as uexp # type: ignore

     
root_path = Path.cwd().parent 

spec_filename   = root_path / 'spectra'
fig_path        = root_path / 'figs'

@dataclass
class GammaPeak:
    """
    Data class to hold measurements for a specific gamma-ray energy peak.

    Parameters
    ----------
    energy : float
        The gamma-ray energy in keV.

    Attributes
    ----------
    energy : float
        The gamma-ray energy in keV.
    times : list[float]
        List of measurement times in seconds since the end of irradiation.
    activities : list[float]
        List of measured activities in Bq.
    uncertainties : list[float]
        List of uncertainties in the measured activities.
    n_measurements : int
        The number of measurements for this peak.
        
    Methods
    -------
    add_measurement(time: float, activity: float, uncertainty: float) -> None
        Add a new measurement for this energy peak.
    
    """
    energy: float
    times: list[float] = field(default_factory=list)
    activities: list[float] = field(default_factory=list)
    uncertainties: list[float] = field(default_factory=list)
    
    def add_measurement(self, time: float, activity: float, uncertainty: float) -> None:
        """
        Add a new measurement for this energy peak.
        
        Parameters
        ----------
        time : float
            The measurement time in seconds since the end of irradiation.
        activity : float
            The measured activity in Bq.
        uncertainty : float
            The uncertainty in the measured activity.
        """
        self.times.append(time)
        self.activities.append(activity)
        self.uncertainties.append(uncertainty)
   
    @property
    def n_measurements(self) -> int:
        """Return the number of measurements for this peak."""
        return len(self.times)

@dataclass
class IsotopeResults:
    """
    Stores analysis results for a single isotope.

    This data class aggregates all measurements for a specific isotope, organized
    by gamma-ray energy peaks. It also stores the optimal parameters from the
    curve fit.

    Parameters
    ----------
    isotope : str
        The identifier for the isotope (e.g., '108AG').
    peaks : list[GammaPeak], optional
        A list of `GammaPeak` objects for this isotope. Initialized to an empty list.
    A0 : UFloat, optional
        The fitted initial activity (A0) at t=0, as a `UFloat` object from the
        `uncertainties` library. Initialized to 0.
    cov : np.ndarray | None, optional
        The covariance matrix from the activity curve fit. Initialized to `None`.

    Attributes
    ----------
    isotope : str
        The identifier for the isotope.
    peaks : list[GammaPeak]
        List of `GammaPeak` objects, each containing measurements for a specific
        gamma-ray energy.
    A0 : UFloat
        The fitted initial activity (A0) at t=0 with its uncertainty.
    cov : np.ndarray | None
        The covariance matrix from the fit.

    Methods
    -------
    get_peak(energy: float, tolerance: float = 0.5) -> GammaPeak | None
        Finds a `GammaPeak` object for a given energy within a tolerance.
    add_or_update_peak(energy: float, time: float, activity: float, uncertainty: float, tolerance: float = 0.5) -> None
        Adds a measurement to an existing peak or creates a new one.
    """
    isotope: str
    peaks: list[GammaPeak] = field(default_factory=list)
    A0: UFloat = ufloat(0, 1e-16)
    cov: np.ndarray | None = None
    
    def _get_peak(self, energy: float, tolerance: float = 0.5) -> GammaPeak | None:
        """
        Find a peak with the given energy within the specified tolerance.
        Returns None if no matching peak is found.
        """
        for peak in self.peaks:
            if abs(peak.energy - energy) <= tolerance:
                return peak
        return None
    
    def add_or_update_peak(self, energy: float, time: float, activity: float, 
                           uncertainty: float, tolerance: float = 0.5) -> None:
        """
        Adds a measurement to an existing peak or creates a new one.

        This method searches for a `GammaPeak` corresponding to the given `energy`
        within a specified `tolerance`. If a matching peak is found, the new
        measurement (time, activity, uncertainty) is added to it. If no
        matching peak is found, a new `GammaPeak` is created and the
        measurement is added to the new peak.

        Parameters
        ----------
        energy : float
            The gamma-ray energy of the measurement in keV.
        time : float
            The measurement time in seconds since the end of irradiation.
        activity : float
            The measured activity in Bq.
        uncertainty : float
            The uncertainty in the measured activity.
        tolerance : float, optional
            The tolerance in keV for matching an existing energy peak, by default 0.5.
        """
        peak = self._get_peak(energy, tolerance)
        if peak is None:
            # Create a new peak
            peak = GammaPeak(energy=energy)
            self.peaks.append(peak)
        
        # Add the measurement to the peak
        peak.add_measurement(time, activity, uncertainty)
    
class SpectrumAnalysis:
    """
    A class for analyzing gamma-ray spectra from irradiated samples.
    
    This class processes spectra from irradiated samples to calculate activities
    of specific isotopes. It extracts peak information, calculates activities,
    and fits decay curves to determine initial activities at the end of irradiation.
    
    Parameters
    ----------
    spec_filename : str or Path
        Path to the spectrum file(s) without the loop number and extension.
        Expected format: 'job{job_number}_Ag{plate_number}_{irradiation_time}min_real{real_time}_loop{num_loops}_'
    Δt_d : float
        Delay time between the end of irradiation and the start of measurement (seconds).
    calibration_source : Path or ci.Calibration, optional
        Either a path to a calibration file or a Calibration object, 
        default is 'calibration.json' in the root path.
        
    Attributes
    ----------
    spec_filename : Path
        Path to the spectrum file(s).
    calib_path : Path or None
        Path to the calibration file if a file path was provided.
    cb : ci.Calibration
        Calibration object for energy and efficiency calibration.
    Δt_d : float
        Delay time between irradiation and measurement (seconds).
    job_specs : dict
        Dictionary containing job specifications extracted from the filename.
    isotopes : list[str]
        List of isotopes to analyze.
    fit_config : dict
        Configuration parameters for peak fitting.
    spectrums : np.ndarray
        Array of Spectrum objects for each measurement loop.
    real_times : np.ndarray
        Array of the actual measurement times for each loop.
    live_times : np.ndarray
        Array of live measurement times for each loop.
    start_times : list[pd.Timestamp]
        List of measurement start times for each loop.
    time_deltas : np.ndarray
        Array of time differences between each measurement and the first measurement.
    isotope_energy : set
        Set of tuples containing (isotope, energy) pairs found in the spectra.
    true_times : np.ndarray
        Array of times since the end of irradiation for each measurement.
    Ag108 : IsotopeResults
        Results for Ag-108 isotope measurements and fits.
    Ag110 : IsotopeResults
        Results for Ag-110 isotope measurements and fits.
    A0_analytical_108 : list
        List of analytical A0 values for Ag-108 measurements.
    A0_analytical_110 : list
        List of analytical A0 values for Ag-110 measurements.
        
    Methods
    -------
    plot_activity(save_fig=True)
        Plot activities for each isotope with different colors for different energy peaks.
    plot_A0_analytical(save_fig=True)
        Plot the analytical A0 values for each isotope and peak.
    A0_func(N_c, λ, ε, I_γ, Δt_c, Δt_d)
        Calculate initial activity analytically.
        
    Internal Methods
    ----------------
    _activity_model(t, λ, A0)
        Model for activity decay.
    _get_job_specs()
        Extract job specifications from the spectrum filename.
    _read_spectrums(job_specs)
        Read spectrum files and extract relevant data.
    _calculate_activities(spectrums)
        Calculate activities for each gamma peak in each spectrum.
    _fit_combined_activity(iso_results)
        Fit activity curve using all peaks for an isotope.
    _plot_isotope_data(ax, iso_results)
        Plot data for a specific isotope on the given axes.
    """



    def __init__(self, spec_filename: str | Path, Δt_d: float, calibration_source: Path | ci.Calibration = root_path / 'calibration.json'):
        # --- Input Validation ---
        if isinstance(spec_filename, str):
            self.spec_filename = Path(spec_filename)
        else:
            self.spec_filename = spec_filename
            
        # Check for valid paths
        spec_path_first_loop = Path(spec_filename + '_000.Spe')
        if not spec_path_first_loop.is_file():
            raise FileNotFoundError(f"First loop of spectrum file not found: {spec_path_first_loop}")
        
        # Check if calibration_file is a Path or a Calibration object
        if isinstance(calibration_source, Path):
            self.calib_path = calibration_source
            if self.calib_path.is_file():
                self.cb = ci.Calibration(self.calib_path)
            else:
                raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")
            
        elif isinstance(calibration_source, ci.Calibration):
            self.calib_path = None
            self.cb = calibration_source
        else:
            raise TypeError("calibration_source must be a Path or a ci.Calibration object")
        
        # --- Initialization ---
        self.Δt_d = Δt_d  # Delay time between irradiation and measurement in seconds
        self.job_specs = self._get_job_specs()
        
        self.isotopes = ['108AG', '110AG']  # Isotopes to analyze
        self.fit_config = {'SNR_min': 3.5, 'dE_511': 9}  # Configuration for fitting
        
        # --- Data Extraction ---
        self.spectrums, self.real_times, self.live_times, self.start_times, self.time_deltas, self.isotope_energy = self._read_spectrums(self.job_specs)

        # --- Activity Calculations ---
        self.Ag108, self.Ag110 = self._calculate_activities(self.spectrums)

        # --- Fitting Activities ---
        self.true_times = self.time_deltas  + self.Δt_d  # Add delay time to real times
        self.Ag108.A0, self.Ag108.cov = self._fit_combined_activity(self.Ag108)
        self.Ag110.A0, self.Ag110.cov = self._fit_combined_activity(self.Ag110)
    

    def _get_job_specs(self):
        """Extract job specifications from the spectrum filename."""
        parts = self.spec_filename.stem.split('_')
        job_specs = {
            'job_number': int(parts[0][3:]),  # Extract job number from 'job1_' -> 1
            'plate_number': int(parts[1][2:]),  # Extract plate number from 'Ag4_' -> 4
            'irradiation_time': int(parts[2][0]), # Extract irradiation time from '1min' -> 1
            'real_time': int(parts[3][4:]),  # Extract real time from 'real10' -> 10
            'num_loops': int(parts[4][4:])  # Extract number of loops from 'loop6' -> 6
            }
        
        return job_specs
    
    def _read_spectrums(self, job_specs: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp], np.ndarray, set]:
        """Read the spectrum files and return a list of Spectrum objects."""
        spectrums      = []
        real_times     = []
        live_times     = [] 
        start_times    = []
        time_deltas    = []
        isotope_energy = set() # ('isotope1', energy1, 'isotope1', energy2, 'isotope2', energy3, ...) 
        
        n_loops = job_specs['num_loops']
        for i in range(n_loops):
            spec_path = self.spec_filename + f'_{i:03d}.Spe'
            spectrum = ci.Spectrum(spec_path)
            spectrum.cb = self.cb
            spectrum.isotopes = self.isotopes
            spectrum.fit_config = self.fit_config
            
            spectrums.append(spectrum)
            real_times.append(spectrum.peaks['real_time'].array[0])  # All peaks have the same real time
            live_times.append(spectrum.peaks['live_time'].array[0])  # All peaks have the same live time
            start_times.append(pd.Timestamp(spectrum.peaks['start_time'].array[0]))  # All peaks have the same start time
            time_deltas.append(pd.Timedelta(start_times[-1] - start_times[0]).total_seconds())
            # Add energies to the set
            isotope_energy.update((zip(spectrum.peaks['isotope'].array, spectrum.peaks['energy'].array)))
            
        real_times  = np.array(real_times)
        live_times  = np.array(live_times)
        spectrums   = np.array(spectrums)
        time_deltas = np.array(time_deltas)
        
        return spectrums, real_times, live_times, start_times, time_deltas, isotope_energy
    
    def _calculate_activities(self, spectrums):
        """Calculate activities for each gamma peak in each spectrum."""
        Ag108 = IsotopeResults('108AG')
        Ag110 = IsotopeResults('110AG')
        self.A0_analytical_108 = []
        self.A0_analytical_110 = []
        # Process each spectrum
        for spec_idx, (spec, time_delta) in enumerate(zip(spectrums, self.time_deltas)):
            # Calculate true time (time since end of irradiation)
            true_time = time_delta + self.Δt_d
            
            # Process each peak in the spectrum
            for E, iso, N_c, unc_N_c, I, unc_I, ε, unc_ε, λ, unc_λ, rt, lt, st in zip(
                    spec.peaks['energy'].array, 
                    spec.peaks['isotope'],
                    spec.peaks['counts'].array, 
                    spec.peaks['unc_counts'].array, 
                    spec.peaks['intensity'].array, 
                    spec.peaks['unc_intensity'].array, 
                    spec.peaks['efficiency'].array, 
                    spec.peaks['unc_efficiency'].array, 
                    spec.peaks['decay_rate'].array, 
                    spec.peaks['unc_decay_rate'].array,
                    spec.peaks['real_time'].array,
                    spec.peaks['live_time'].array,
                    spec.peaks['start_time'].array):
                
                # Get decay constant from isotope library
                λ, unc_λ = ci.Isotope(iso).decay_const(unc=True) # type: ignore
                
                λ_u   = ufloat(λ, unc_λ)
                N_c_u = ufloat(N_c, unc_N_c)
                I_u   = ufloat(I, unc_I)
                ε_u   = ufloat(ε, unc_ε)
                
                # Calculate activity for this peak
                # TODO: Is this the correct formula?
                A = (N_c_u * λ_u) / (ε_u * I_u * (1 - uexp(-λ_u * lt))) # type: ignore
                A0_approx = self.A0_func(N_c_u, λ_u, ε_u, I_u, lt, true_time)
                
                # Add the measurement to the appropriate isotope result
                if iso == '108AG':
                    Ag108.add_or_update_peak(
                        energy=E, 
                        time=true_time,
                        activity=A.nominal_value,
                        uncertainty=A.std_dev
                    )
                    self.A0_analytical_108.append(A0_approx.nominal_value)
                elif iso == '110AG':
                    Ag110.add_or_update_peak(
                        energy=E, 
                        time=true_time,
                        activity=A.nominal_value,
                        uncertainty=A.std_dev
                    )
                    self.A0_analytical_110.append(A0_approx.nominal_value)
                else:
                    print(f"Warning: Unrecognized isotope {iso} in spectrum {spec_idx}. Skipping.")
        
        return Ag108, Ag110
                    
    def _fit_combined_activity(self, iso_results: IsotopeResults):
        """Fit activity curve using all peaks for an isotope."""
        # Collect all times, activities, and uncertainties across all peaks
        all_times = []
        all_activities = []
        all_uncertainties = []

        for peak in iso_results.peaks:
            all_times.extend(peak.times)
            all_activities.extend(peak.activities)
            all_uncertainties.extend(peak.uncertainties)
            
        if len(all_times) == 0:
            raise ValueError(f"No measurements found for isotope {iso_results.isotope}. Cannot fit activity curve.")

        # Sorts all data by time
        sorted_data = sorted(zip(all_times, all_activities, all_uncertainties), key=lambda x: x[0])
        all_times, all_activities, all_uncertainties = zip(*sorted_data)
        
        # Fitting the data
        λ = ci.Isotope(iso_results.isotope).decay_const()
        try:
            params, cov = curve_fit(
                lambda t, A0: self._activity_model(t, λ, A0),
                all_times, 
                all_activities, 
                p0=[max(all_activities)], 
                sigma=all_uncertainties, 
                absolute_sigma=True
            )
            A0 = ufloat(params[0], np.sqrt(cov[0, 0]))
            return A0, cov
        
        except ValueError:
            raise ValueError(f'Error fitting activity as measurement data contains NaN, inf.\nTimes: {all_times}\nActicvities: {all_activities}')

        except RuntimeError:
            raise RuntimeError(f'Failed least squares fit for isotope {iso_results.isotope}. Check data quality or fitting parameters.')
            
    def _plot_isotope_data(self, ax, iso_results):
        """Plot data for a specific isotope."""
        # Check iso_results for possible issues
        
        # TODO: Evalute using differnt colors
        # Use a different color for each energy peak
        # colors = plt.cm.tab10(np.linspace(0, 1, len(iso_results.peaks)))
        # colors = plt.cm.brg(np.linspace(0, 1, len(iso_results.peaks)))
        # colors = plt.cm.Set1(np.linspace(0, 1, len(iso_results.peaks)))
        
        # Plot each energy peak with its own color
        for i, peak in enumerate(iso_results.peaks):
            # if peak.n_measurements > 0:
            label = f"{peak.energy:.1f} keV"
            ax.errorbar(
                peak.times, 
                peak.activities, 
                yerr=peak.uncertainties, 
                fmt='o:', 
                capsize=5,
                # color=colors[i],
                label=label,
                alpha=1-.2*i,  # Decrease alpha for each peak
            )
        
        # Get time range for plotting
        all_times = []
        for peak in iso_results.peaks:
            all_times.extend(peak.times)
        
        # if all_times:
        t_min = min(all_times)
        t_max = max(all_times)
        plot_times = np.linspace(0, t_max * 1.1, 100)
        
        # Get decay constant
        λ = ci.Isotope(iso_results.isotope).decay_const()
        
        # Plot fit line
        fit_line = self._activity_model(plot_times, λ, iso_results.A0.nominal_value)
        ax.plot(plot_times, fit_line, 'k-', 
                label=f'Combined Fit')
        
        # Plot confidence band
        # if iso_results.cov is not None:
        # Calculate uncertainty in fit
        fit_unc = np.abs(fit_line * (iso_results.A0.std_dev / iso_results.A0.nominal_value))
        ax.fill_between(plot_times, 
                    fit_line - fit_unc, 
                    fit_line + fit_unc, 
                    color='gray', alpha=0.3, label='1σ Confidence Band')

    
        # formatted_iso_name = rf'$^{iso_results.isotope[:3]}$Ag'
        formatted_iso_name = rf'$^{{{iso_results.isotope[:3]}}}$Ag'
        ax.set_title(formatted_iso_name + f'\nA₀={iso_results.A0:.2uP} Bq')
        ax.set_xlabel('Time since irradiation end [s]')
        ax.set_ylabel('Activity [Bq]')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_activity(self, save_fig: bool = True):
        """Plot activities for each isotope, with different colors for different energy peaks."""
        plt.figure()
        # Plot 108Ag data
        ax1 = plt.subplot(1, 2, 1)
        self._plot_isotope_data(ax1, self.Ag108)
        
        # Plot 110Ag data
        ax2 = plt.subplot(1, 2, 2)
        self._plot_isotope_data(ax2, self.Ag110)
        
        plt.suptitle(f'Activity Analysis for {self.spec_filename.stem}')
        plt.tight_layout()
        
        if save_fig:
            path = fig_path / f'{self.spec_filename.stem}_activity_analysis'
            plt.savefig(path.with_suffix('.pdf'))
            plt.savefig(path.with_suffix('.png'))
        plt.show()
    
    def plot_A0_analytical(self, save_fig: bool = True):
        """Plot the analytical A0 values for each isotope and peak."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # --- Plot for 108Ag ---
        for peak in self.Ag108.peaks:
            # Filter the A0 values for the current peak energy
            a0_values = [a0 for e, a0 in zip(self.A0_analytical_108, self.A0_analytical_108) if abs(e - peak.energy) < 0.5]
            ax1.plot(range(len(a0_values)), a0_values, 'o--', label=f'{peak.energy:.1f} keV')
        
        formatted_iso_name_108 = rf'$^{{{self.Ag108.isotope[:3]}}}$Ag'
        ax1.set_title(f'Analytical A₀ for {formatted_iso_name_108}')
        ax1.set_xlabel('Measurement Index')
        ax1.set_ylabel('Analytical A₀ [Bq]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot for 110Ag ---
        for peak in self.Ag110.peaks:
            # Filter the A0 values for the current peak energy
            a0_values = [a0 for e, a0 in zip(self.A0_analytical_110, self.A0_analytical_110) if abs(e - peak.energy) < 0.5]
            ax2.plot(range(len(a0_values)), a0_values, 'o--', label=f'{peak.energy:.1f} keV')

        formatted_iso_name_110 = rf'$^{{{self.Ag110.isotope[:3]}}}$Ag'
        ax2.set_title(f'Analytical A₀ for {formatted_iso_name_110}')
        ax2.set_xlabel('Measurement Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Analytical A₀ Calculation for {self.spec_filename.stem}')
        plt.tight_layout(rect=(0., 0.03, 1., 0.95))

        if save_fig:
            path = fig_path / f'{self.spec_filename.stem}_A0_analytical'
            plt.savefig(path.with_suffix('.pdf'))
            plt.savefig(path.with_suffix('.png'))
        plt.show()
        
        
    def _activity_model(self, t, λ, A0):
        """Model for activity decay."""
        return A0*np.exp(-λ*t)
    
    # TODO: Contemplate discarding analytical A0 calculation
    def A0_func(self, N_c, λ, ε, I_γ, Δt_c, Δt_d):
        return (N_c * λ) / (ε * I_γ * (1 - uexp(-λ * Δt_c))) * uexp(λ * Δt_d)