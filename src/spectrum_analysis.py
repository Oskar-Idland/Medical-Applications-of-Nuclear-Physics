import pathlib
import curie as ci
import numpy as np
import pandas as pd
from path import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.axes import Axes
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from uncertainties import ufloat, UFloat
from uncertainties.umath import exp as uexp  # type: ignore


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
    `add_measurement`(time: float, activity: float, uncertainty: float) -> None:
        Add a new measurement for this energy peak.
    
    """
    energy: float
    times: list[float] = field(default_factory=list)
    activities: list[float] = field(default_factory=list)
    uncertainties: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate energy value after initialization."""
        if self.energy <= 0:
            raise ValueError(f"Energy must be positive, got {self.energy}")
        
    def add_measurement(self,
                        time: float,
                        activity: float,
                        uncertainty: float) -> None:
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
    cov : NDArray | None, optional
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
    cov : NDArray | None
        The covariance matrix from the fit.

    Methods
    -------
    `get_peak`(energy: float, tolerance: float = 0.5) -> GammaPeak | None:
        Finds a `GammaPeak` object for a given energy within a tolerance.
    `add_or_update_peak`(energy: float, time: float, activity: float, uncertainty: float, tolerance: float = 0.5) -> None:
        Adds a measurement to an existing peak or creates a new one.
    """
    isotope: str
    peaks: list[GammaPeak] = field(default_factory=list)
    A0: UFloat = ufloat(0, 1e-16)
    cov: NDArray | None = None
    
    def _get_peak(self,
                  energy: float,
                  tolerance: float = 0.5) -> GammaPeak | None:
        """
        Find a peak with the given energy within the specified tolerance.
        Returns None if no matching peak is found.
        """
        for peak in self.peaks:
            if abs(peak.energy - energy) <= tolerance:
                return peak
        return None
    
    def add_or_update_peak(self,
                           energy: float,
                           time: float,
                           activity: float, 
                           uncertainty: float,
                           tolerance: float = 0.5) -> None:
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
    root_path : Path
        Root directory for the project.
    spec_base_path : Path
        Directory containing spectrum files.
    fig_path : Path
        Directory for saving figures.
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
    spectrums : NDArray
        Array of Spectrum objects for each measurement loop.
    real_times : NDArray
        Array of the actual measurement times for each loop.
    live_times : NDArray
        Array of live measurement times for each loop.
    start_times : list[pd.Timestamp]
        List of measurement start times for each loop.
    time_deltas : NDArray
        Array of time differences between each measurement and the first measurement.
    isotope_energy : set
        Set of tuples containing (isotope, energy) pairs found in the spectra.
    true_times : NDArray
        Array of times since the end of irradiation for each measurement.
    Ag108 : IsotopeResults
        Results for Ag-108 isotope measurements and fits.
    Ag110 : IsotopeResults
        Results for Ag-110 isotope measurements and fits.
    A0_analytical_108 : list[float]
        List of analytical A0 values for Ag-108 measurements.
    A0_analytical_110 : list[float]
        List of analytical A0 values for Ag-110 measurements.
    A0_analytical_energies_108 : list[float]
        List of energies corresponding to A0_analytical_108 measurements.
    A0_analytical_energies_110 : list[float]
        List of energies corresponding to A0_analytical_110 measurements.

    Methods
    -------
    `plot_activity`(save_fig: bool = True)
        Plot activities for each isotope with different colors for different energy peaks.
    `plot_A0_analytical`(save_fig: bool = True)
        Plot the analytical A0 values for each isotope and peak.
    `A0_func`(N_c: UFloat, λ: UFloat, ε: UFloat, I_γ: UFloat, Δt_c: float, Δt_d: float) -> UFloat
        Calculate initial activity analytically.

    Internal Methods
    ----------------
    `_activity_model`(t: NDArray, λ: float, A0: float) -> NDArray
        Model for activity decay.
    `_get_job_specs`() -> dict
        Extract job specifications from the spectrum filename.
    `_read_spectrums`(job_specs: dict[str, int]) -> tuple[NDArray, NDArray, NDArray, list[pd.Timestamp], NDArray, set]
        Read spectrum files and extract relevant data.
    `_calculate_activities`(spectrums: NDArray) -> tuple[IsotopeResults, IsotopeResults]
        Calculate activities for each gamma peak in each spectrum.
    `_fit_combined_activity`(iso_results: IsotopeResults) -> tuple[UFloat, NDArray]
        Fit activity curve using all peaks for an isotope.
    `_plot_isotope_data`(ax: plt.Axes, iso_results: IsotopeResults) -> None
        Plot data for a specific isotope on the given axes.
    """

    def __init__(self,
                 spec_filename: str | Path,
                 Δt_d: float,
                 calibration_source: Path | ci.Calibration | None = None,
                 root_path: Path | None = None,
                 spec_dir: str = 'spectra',
                 fig_dir: str = 'figs') -> None:
        
        # Setup paths
        self.root_path = root_path or Path(__file__).parent.parent
        self.spec_base_path = self.root_path / spec_dir
        self.fig_path = self.root_path / fig_dir

        # Setup spectrum filename
        self.spec_filename = Path(spec_filename)

        # Validate spectrum file existence
        spec_path_first_loop = self.spec_filename.with_name(self.spec_filename.name + '_000.Spe')
        if not spec_path_first_loop.is_file():
            raise FileNotFoundError(f"First loop of spectrum file not found: {spec_path_first_loop}")
        
        # Setup calibration
        calibration_source = calibration_source or (self.root_path / 'calibration.json')

        if isinstance(calibration_source, ci.Calibration):
            self.calib_path = None
            self.cb = calibration_source
        else:
            self.calib_path = calibration_source
            if not self.calib_path.is_file():
                raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")
            self.cb = ci.Calibration(self.calib_path)
        
        # Core parameters
        self.Δt_d = Δt_d  # Delay time between irradiation and measurement (s)
        self.isotopes = ['108AG', '110AG']  # Isotopes to analyze
        self.fit_config = {'SNR_min': 3.5, 'dE_511': 9}  # Configuration for fitting

        # Extract job info and run analysis
        self.job_specs = self._get_job_specs()
        self.spectrums, self.real_times, self.live_times, self.start_times, self.time_deltas, self.isotope_energy = self._read_spectrums(self.job_specs)
        self.Ag108, self.Ag110 = self._calculate_activities(self.spectrums)
        self.true_times = self.time_deltas  + self.Δt_d
        self.Ag108.A0, self.Ag108.cov = self._fit_combined_activity(self.Ag108)
        self.Ag110.A0, self.Ag110.cov = self._fit_combined_activity(self.Ag110)
    

    def _get_job_specs(self) -> dict[str, int]:
        """
        Extracts job specifications from the spectrum filename.

        The filename is expected to follow the format: 'jobX_AgY_Zmin_realW_loopV', where:
            - X: job number (e.g., 'job1')
            - Y: plate number (e.g., 'Ag4')
            - Z: irradiation time in minutes (e.g., '1min')
            - W: real time in minutes (e.g., 'real10')
            - V: number of loops (e.g., 'loop6')
            
        Returns
        -------
        dict[str, int]
            A dictionary containing the extracted job specifications with the following keys:
            - 'job_number' (int): The job number.
            - 'plate_number' (int): The plate number.
            - 'irradiation_time' (int): The irradiation time in minutes.
            - 'real_time' (int): The real time in minutes.
            - 'num_loops' (int): The number of loops.
        """
        parts = self.spec_filename.stem.split('_')

        return {
            'job_number': int(parts[0][3:]),  
            'plate_number': int(parts[1][2:]), 
            'irradiation_time': int(parts[2][0]), 
            'real_time': int(parts[3][4:]),  
            'num_loops': int(parts[4][4:])
        }

    def _create_configured_spectrum(self,
                                    loop_index: int) -> tuple[ci.Spectrum, pathlib.Path]:
        """
        Create and configure a Spectrum object for a specific measurement loop.

        Parameters
        ----------
        loop_index : int
            The index of the measurement loop (starting from 0).

        Returns
        -------
        tuple[ci.Spectrum, pathlib.Path]
            A tuple containing the configured Spectrum object and its file path.
        """
        spec_path = self.spec_filename.with_name(f"{self.spec_filename.stem}_{loop_index:03d}.Spe")
        spectrum = ci.Spectrum(spec_path)
        spectrum.cb = self.cb
        spectrum.isotopes = self.isotopes
        spectrum.fit_config = self.fit_config
        return spectrum, spec_path

    def _read_spectrums(self,
                        job_specs: dict[str, int]) -> tuple[NDArray, NDArray, NDArray, list[pd.Timestamp], NDArray, set]:
        """
        Read the spectrum files and return arrays of Spectrum objects and associated measurement data.

        This method reads all spectrum files for each measurement loop, extracts peak data,
        and computes arrays for real times, live times, start times, time deltas, and a set
        of all unique (isotope, energy) pairs found in the spectra.

        Parameters
        ----------
        job_specs : dict[str, int]
            Dictionary containing job specifications with keys for num_loops.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, list[pd.Timestamp], NDArray, set]
            Containing:
            - spectrums: Array of Spectrum objects for each measurement loop
            - real_times: Array of real measurement times for each loop
            - live_times: Array of live measurement times for each loop
            - start_times: List of measurement start timestamps for each loop
            - time_deltas: Array of time differences (in seconds) from the first measurement
            - isotope_energy: Set of (isotope, energy) tuples found in all spectra
        """
        spectrums = []
        real_times = []
        live_times = []
        start_times = []
        time_deltas = []
        isotope_energy = set()  # Set of (isotope, energy) tuples found in all spectra

        # First, collect all valid spectrum-peak pairs
        valid_data = []
        num_loops = job_specs['num_loops']
        for i in range(num_loops):
            spectrum, spec_path = self._create_configured_spectrum(i)
            peaks = spectrum.peaks
            
            if peaks is None:
                print(f"Warning: No peaks found in spectrum {spec_path}. Skipping.")
                continue
                
            valid_data.append((spectrum, peaks))
        
        # Extract data using list comprehensions
        spectrums = np.array([spectrum for spectrum, _ in valid_data])
        real_times = np.array([peaks['real_time'].array[0] for _, peaks in valid_data])
        live_times = np.array([peaks['live_time'].array[0] for _, peaks in valid_data])
        start_times = [pd.Timestamp(peaks['start_time'].array[0]) for _, peaks in valid_data]
        
        # Calculate time deltas
        time_deltas = np.array([0] + [
            pd.Timedelta(start_times[i] - start_times[i-1]).total_seconds() 
            for i in range(1, len(start_times))
        ])

        # Collect all unique isotope-energy pairs
        for _, peaks in valid_data:
            isotope_energy.update(zip(peaks['isotope'].array, peaks['energy'].array))
        
        return spectrums, real_times, live_times, start_times, time_deltas, isotope_energy
    
    def _process_peak(self,
                      peak_data: dict,
                      true_time: float) -> tuple[str, float, UFloat, UFloat]:
        """
        Processes peak data to calculate the activity and approximate initial activity (A0) for a given isotope peak.
        
        Parameters
        ----------
        peak_data : dict
            Dictionary containing peak information with the following keys:
                - E (float): Energy of the peak.
                - iso (str): Isotope identifier.
                - N_c (float): Net counts.
                - unc_N_c (float): Uncertainty in net counts.
                - I (float): Intensity of the gamma line.
                - unc_I (float): Uncertainty in intensity.
                - ε (float): Detection efficiency.
                - unc_ε (float): Uncertainty in efficiency.
                - lt (float): Live time of the measurement.
        true_time : float
            The true time of the measurement.

        Returns
        -------
        tuple[str, float, UFloat, UFloat]
            Containing:
                - iso (str): Isotope identifier.
                - E (float): Energy of the peak.
                            - A (UFloat): Calculated activity with uncertainty.
                            - A0_approx (UFloat): Approximated initial activity with uncertainty.
        """
        # Extract peak data
        E, iso, N_c, unc_N_c, I, unc_I, ε, unc_ε, lt = peak_data.values()

        # Get decay constant from isotope library
        λ_u = ci.Isotope(iso).decay_const(unc=True)
        if isinstance(λ_u, tuple):
            λ_u = ufloat(λ_u[0], λ_u[1])

        # Create ufloat objects
        N_c_u = ufloat(N_c, unc_N_c)
        I_u = ufloat(I, unc_I)
        ε_u = ufloat(ε, unc_ε)

        # Calculate activity and A0
        A = (N_c_u * λ_u) / (ε_u * I_u * (1 - uexp(-λ_u * lt)))  # type: ignore
        A0_approx = self.A0_func(N_c_u, λ_u, ε_u, I_u, lt, true_time)  # type: ignore

        return iso, E, A, A0_approx

    def _calculate_activities(self, 
                              spectrums: NDArray) -> tuple[IsotopeResults, IsotopeResults]:
        """
        Calculate activities for each gamma peak in each spectrum.
        
        Parameters
        ----------
        spectrums : NDArray
            Array of Spectrum objects to analyze.
            
        Returns
        -------
        tuple[IsotopeResults, IsotopeResults]
            Tuple containing IsotopeResults for Ag-108 and Ag-110.
        """
        # Initialize results dictionary
        isotope_results = {
            '108AG': IsotopeResults('108AG'),
            '110AG': IsotopeResults('110AG')
        }
        analytical_A0 = {'108AG': [], '110AG': []}
        analytical_energies = {'108AG': [], '110AG': []}

        # Process each spectrum
        for spec_idx, (spec, time_delta) in enumerate(zip(spectrums, self.time_deltas)):
            true_time = time_delta + self.Δt_d
            
            # Process each peak in the spectrum
            for E, iso, N_c, unc_N_c, I, unc_I, ε, unc_ε, lt in zip(
                    spec.peaks['energy'].array, 
                    spec.peaks['isotope'],
                    spec.peaks['counts'].array, 
                    spec.peaks['unc_counts'].array, 
                    spec.peaks['intensity'].array, 
                    spec.peaks['unc_intensity'].array, 
                    spec.peaks['efficiency'].array, 
                    spec.peaks['unc_efficiency'].array, 
                    spec.peaks['live_time'].array,
            ):
                
                # Create peak data dictionary
                peak_data = {
                    'E': E, 'iso': iso, 'N_c': N_c, 'unc_N_c': unc_N_c,
                    'I': I, 'unc_I': unc_I, 'ε': ε, 'unc_ε': unc_ε, 'lt': lt
                }

                iso, E, A, A0_approx = self._process_peak(peak_data, true_time)

                # Add to the appropriate isotope result
                if iso in isotope_results:
                    isotope_results[iso].add_or_update_peak(
                        energy=E, 
                        time=true_time,
                        activity=A.nominal_value,
                        uncertainty=A.std_dev
                    )
                    analytical_A0[iso].append(A0_approx.nominal_value)
                    analytical_energies[iso].append(E)
                    
                else:
                    print(f"Warning: Unrecognized isotope {iso} in spectrum {spec_idx}. Skipping.")

        self.A0_analytical_108 = analytical_A0['108AG']
        self.A0_analytical_110 = analytical_A0['110AG']
        self.A0_analytical_energies_108 = analytical_energies['108AG']
        self.A0_analytical_energies_110 = analytical_energies['110AG']

        return isotope_results['108AG'], isotope_results['110AG']
               
    def _fit_combined_activity(self, 
                               iso_results: IsotopeResults) -> tuple[UFloat, NDArray]:
        """
        Fit the activity decay curve using all measurements from all peaks for a given isotope.

        Collects times, activities, and uncertainties from all `GammaPeak` objects in `IsotopeResults`, and
        fits the decay model A(t) = A0 * exp(-λ * t) using non-linear least squares.

        Parameters
        ----------
        iso_results : IsotopeResults
            The IsotopeResults object containing all GammaPeak measurements for the isotope.

        Returns
        -------
        tuple[UFloat, NDArray]
            Containing:
            - A0 (UFloat): The fitted initial activity at t=0 with uncertainty.
            - cov (NDArray): The covariance matrix from the curve fit.

        Raises
        ------
        ValueError
            If no measurements are found for the isotope or if the fit fails due to invalid data.
        RuntimeError
            If the optimization fails to converge.
        """
        # Collect all times, activities, and uncertainties from all peaks
        all_times = np.array([time for peak in iso_results.peaks for time in peak.times])
        all_activities = np.array([activity for peak in iso_results.peaks for activity in peak.activities])
        all_uncertainties = np.array([unc for peak in iso_results.peaks for unc in peak.uncertainties])
                    
        if len(all_times) == 0:
            raise ValueError(f"No measurements found for isotope {iso_results.isotope}. Cannot fit activity curve.")

        # Sorts all data by time
        sort_indices = np.argsort(all_times)
        all_times = all_times[sort_indices]
        all_activities = all_activities[sort_indices]
        all_uncertainties = all_uncertainties[sort_indices]

        # Fitting the data and estimating A0
        λ = ci.Isotope(iso_results.isotope).decay_const()

        max_activity = max(all_activities)
        min_time = min(all_times)
        estimated_A0 = max_activity * np.exp(λ * min_time)

        try:
            params, cov = curve_fit(
                lambda t, A0: self._activity_model(t, λ, A0),  # type: ignore
                all_times,
                all_activities,
                p0=[estimated_A0],
                bounds=([0], [np.inf]),
                sigma=all_uncertainties, 
                absolute_sigma=True
            )
            A0 = ufloat(params[0], np.sqrt(cov[0, 0]))
            return A0, cov
        
        except ValueError as e:
            if "NaN" in str(e) or "inf" in str(e):
                raise ValueError(f"Invalid data for {iso_results.isotope}: {str(e)}")
            else:
                raise ValueError(f"Curve fitting failed for {iso_results.isotope}: {str(e)}")

        except RuntimeError as e:
            raise RuntimeError(f"Optimization failed for {iso_results.isotope}: {str(e)}. "
                               f"Try different initial parameters or check data quality. Error: {str(e)}")

    def _plot_isotope_data(self, 
                           ax: Axes, 
                           iso_results: IsotopeResults) -> None:
        """
        Plot activity data and fit for a specific isotope.

        This method visualizes the measured activities for each gamma-ray energy peak of the given isotope,
        including error bars for uncertainties. Each energy peak is plotted with a distinct color.
        The combined exponential decay fit (A(t) = A₀·exp(-λ·t)) is overlaid, along with a 1σ confidence band.
        The plot includes a legend, axis labels, and a title showing the isotope and fitted A₀ value.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        iso_results : IsotopeResults
            The results object containing all peaks and fit data for the isotope.
        """
        # Check if there are any peaks to plot
        if not iso_results.peaks:
            print(f"No peaks found for {iso_results.isotope}")
            return
        
        # Check if A0 is valid
        if iso_results.A0.nominal_value <= 0:
            print(f"Warning: Invalid A0 value for {iso_results.isotope}: {iso_results.A0}")

        # Check if some peaks were excluded
        excluded_peaks = [peak for peak in iso_results.peaks if peak.n_measurements == 0]
        if excluded_peaks:
            print(f"Warning: Found {len(excluded_peaks)} peaks excluded from analysis for {iso_results.isotope} (likely below SNR threshold)")

        # Use a different color for each energy peak
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(iso_results.peaks)))
        
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
                color=colors[i],
                label=label,
                alpha=1-.2*i,  # Decrease alpha for each peak
            )
        
        # Get time range for plotting
        all_times = [time for peak in iso_results.peaks for time in peak.times]

        if not all_times:
            print(f"Warning: No time data found for {iso_results.isotope}")
            return

        t_min, t_max = min(all_times), max(all_times)
        plot_times = np.linspace(max(0, t_min - 0.1), t_max * 1.1, 100)

        # Get decay constant
        λ = ci.Isotope(iso_results.isotope).decay_const() 
        
        # Plot fit line
        fit_line = self._activity_model(plot_times, λ, iso_results.A0.nominal_value)  # type: ignore
        ax.plot(plot_times, fit_line, "k-", label=f"Combined Fit")
        
        # Plot confidence band
        fit_unc = np.abs(fit_line * (iso_results.A0.std_dev / iso_results.A0.nominal_value))
        ax.fill_between(plot_times, 
                    fit_line - fit_unc, 
                    fit_line + fit_unc, 
                    color="gray", alpha=0.3, label="1σ Confidence Band")

        formatted_iso_name = rf"$^{{{iso_results.isotope[:3]}}}$Ag"
        ax.set_title(formatted_iso_name + f"\nA₀={iso_results.A0:.2uP} Bq")
        ax.set_xlabel("Time since irradiation end [s]")
        ax.set_ylabel("Activity [Bq]")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_activity(self, 
                      save_fig: bool = True) -> None:
        """
        Plot activities for each isotope, with different colors for different energy peaks.
        
        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure to file, by default True.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # Plot isotope data
        self._plot_isotope_data(ax1, self.Ag108)
        self._plot_isotope_data(ax2, self.Ag110)

        plt.suptitle(f"Activity Analysis for {self.spec_filename.stem}")
        plt.tight_layout()
        
        if save_fig:
            path = self.fig_path / "activity_analysis" / f"{self.spec_filename.stem}"
            path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(path.with_suffix(".pdf"))
            plt.savefig(path.with_suffix(".png"))
        plt.show()

    def _plot_analytical_A0_for_isotope(self,
                                         ax: Axes,
                                         isotope_results: IsotopeResults,
                                         analytical_A0_list: list[float],
                                         analytical_energies_list: list[float]) -> None:
        """
        Plot the analytical A0 values for a specific isotope.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        isotope_results : IsotopeResults
            The results object containing all peaks and fit data for the isotope.
        analytical_A0_list : list[float]
            List of analytical A0 values for the isotope.
        analytical_energies_list : list[float]
            List of energies corresponding to the analytical A0 values.
        """
        for peak in isotope_results.peaks:
            A0_values = [A0 for E, A0 in zip(analytical_energies_list, analytical_A0_list) if abs(E - peak.energy) < 0.5]
            ax.plot(range(len(A0_values)), A0_values, "o--", label=f"{peak.energy:.1f} keV")
        
        formatted_name = rf"$^{{{isotope_results.isotope[:3]}}}$Ag"
        ax.set_title(f"Analytical A₀ for {formatted_name}")
        ax.set_xlabel("Measurement Index")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_A0_analytical(self, 
                           save_fig: bool = True) -> None:
        """
        Plot the analytical A0 values for each isotope and peak.
        
        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure to file, by default True.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # Use the helper method for both isotopes
        self._plot_analytical_A0_for_isotope(ax1, self.Ag108, self.A0_analytical_108, self.A0_analytical_energies_108)
        self._plot_analytical_A0_for_isotope(ax2, self.Ag110, self.A0_analytical_110, self.A0_analytical_energies_110)

        #Add y-axis label to the first subplot
        ax1.set_ylabel("Analytical A₀ [Bq]")

        plt.suptitle(f"Analytical A₀ Calculation for {self.spec_filename.stem}")
        plt.tight_layout(rect=(0., 0.03, 1., 0.95))

        if save_fig:
            path = self.fig_path / f"{self.spec_filename.stem}_A0_analytical"
            plt.savefig(path.with_suffix(".pdf"))
            plt.savefig(path.with_suffix(".png"))
        plt.show()
        
    def _activity_model(self, 
                        t: NDArray, 
                        λ: float, 
                        A0: float) -> NDArray:
        """
        Exponential decay model for radioactive activity: A(t) = A₀ · exp(-λt).
    
        Parameters
        ----------
        t : NDArray
            Time values since end of irradiation [s].
        λ : float
            Decay constant [s⁻¹].
        A0 : float
            Initial activity at t=0 [Bq].
            
        Returns
        -------
        NDArray
            Activity values at given times [Bq].
        """
        return A0 * np.exp(-λ * t)
    
    def A0_func(self, 
                N_c: UFloat, 
                λ: UFloat, 
                ε: UFloat, 
                I_γ: UFloat, 
                Δt_c: float, 
                Δt_d: float) -> UFloat:
        """
        Calculate initial activity analytically from gamma-ray spectroscopy measurements.
        
        Uses the formula:
        A₀ = (N_c · λ) / [ε · I_γ · (1 - exp(-λ · Δt_c)) · exp(-λ · Δt_d)]
        
        Parameters
        ----------
        N_c : UFloat
            Net counts with uncertainty.
        λ : UFloat
            Decay constant with uncertainty [s⁻¹].
        ε : UFloat
            Detection efficiency (dimensionless) with uncertainty.
        I_γ : UFloat
            Gamma-ray intensity (branching ratio) with uncertainty.
        Δt_c : float
            Count time (live time) [s].
        Δt_d : float
            Delay time since end of irradiation [s].
            
        Returns
        -------
        UFloat
            Initial activity at end of irradiation with uncertainty [Bq].
        """
        count_rate_correction = (1 - uexp(-λ * Δt_c))  # type: ignore
        decay_correction = uexp(-λ * Δt_d)  # type: ignore
        return (N_c * λ) / (ε * I_γ * count_rate_correction * decay_correction)  # type: ignore