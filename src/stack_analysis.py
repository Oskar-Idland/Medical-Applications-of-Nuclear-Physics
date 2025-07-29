import curie as ci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numpy import float64, int64
from collections.abc import Callable
from scipy.constants import Avogadro
from matplotlib.figure import Figure

class StackAnalysis:
    """
    A class for analyzing irradiated target stacks to determine optimal counting times for gamma-ray spectroscopy.

    This class handles:
    - Calculation of isotope activities from cross-sections and irradiation parameters.
    - Determination of counting times required to achieve statistical precision.
    - Analysis of gamma-ray lines with consideration for detector efficiency and decay timing.
    - Visualization of counting time dependencies on delay times.

    Attributes
    ----------
    stack : ci.Stack
        The stack object containing target materials and their areal densities.
    efficiency_func : Callable
        Function that returns detector efficiency as a function of gamma-ray energy (MeV).
    proton_flux : float
        Incident proton flux in protons/cm²/s.
    irradiation_time : float
        Duration of irradiation in seconds.

    Methods
    -------
    `__init__(stack: ci.Stack, effciency_func: Callable, proton_flux: float = 6.24e11, irradiation_time: float = 60*60) -> None`:
        Initialize the StackAnalysis object with stack configuration and irradiation parameters.
    `analyze(products: dict[str, pd.DataFrame], t_d: NDArray[float64|int64] = np.arange(0, 60*60), t_max: float = 60*60, min_intensity: float = 10.0, dE_511: float = 0.1, silent: bool = True) -> pd.DataFrame`:
        Analyze the stack to calculate counting times for gamma lines from each foil across specified delay times.
    `plot(counting_times: pd.DataFrame, target: str | None = None, n_gammas: int = 5, title: str = '') -> Figure`:
        Generate plots of counting times versus delay times for the most favorable gamma lines.
    `_A0(σ: float | NDArray[float64], N_T: int | NDArray[int64], Φ: float | NDArray[float64], t_irr: float | NDArray[float64], λ: float | NDArray[float64]) -> NDArray[float64]`:
        Calculate initial activity from nuclear reaction parameters and decay constants.
    `_counting_time(A0: float | NDArray[float64], I_γ: float | NDArray[float64], t_d: float | NDArray[float64|int64], ε_γ: float | NDArray[float64], λ: float | NDArray[float64], N_c: int | NDArray[int64] = 10_000) -> float | NDArray[float64]`:
        Determine counting time required to achieve specified count statistics as function of delay time.
    """

    # Unit conversion constants
    MILLIBARN_TO_CM2 = 1e-27    # mb -> cm^2
    GRAM_TO_KG = 1e-3           # g -> kg
    PERCENT_TO_FRACTION = 0.01  # % -> fraction

    stack: ci.Stack
    efficiency_func: Callable
    proton_flux: float
    irradiation_time: float

    def __init__(self,
                 stack: ci.Stack,
                 efficiency_func: Callable,
                 proton_flux: float = 6.24e11,
                 irradiation_time: float = 60*60):
        """
        Initializes the StackAnalysis class.
        
        Parameters
        ----------
        stack: ci.Stack
            The stack object containing the materials and their properties.
        proton_flux: float, optional
            The proton flux in protons/cm^2/s. Defaults to 6.24e11.
        irradiation_time: float, optional
            The irradiation time in seconds. Defaults to 60*60 (1 hour).
        """
        self.stack = stack
        self.efficiency_func = efficiency_func
        self.proton_flux = proton_flux
        self.irradiation_time = irradiation_time
    
    def analyze(self, 
                products: dict[str, pd.DataFrame],
                t_d: NDArray[float64|int64] = np.arange(0, 60*60),
                t_max: float = 60*60,
                min_intensity: float = 10.0,
                dE_511: float = 0.1,
                silent: bool = True) -> pd.DataFrame:
        """
        Analyzes the stack by calculating counting times for gamma lines from each foil.

        Parameters
        ----------
        products : dict[str, pd.DataFrame]
            Dictionary mapping target element symbols to DataFrames containing product isotope data.
            Each DataFrame should have columns 'Name', 'E', and 'Cs' where:
            - 'Name' is the isotope name (e.g., '67Ga')
            - 'E' is an array of energy points (MeV)
            - 'Cs' is an array of cross-sections (mb) corresponding to each energy point
        t_d : NDArray[float64 | int64], optional
            Array of delay times in seconds for which to calculate counting times.
            Default is np.arange(0, 60*60).
        t_max : float, optional
            Maximum allowed counting time in seconds. Isotopes requiring longer counting times
            will be skipped. Default is 60*60 (1 hour).
        min_intensity : float, optional
            Minimum gamma-ray intensity (%) to consider. Default is 10.0.
        dE_511 : float, optional
            Energy window (MeV) around 511 keV to exclude when selecting gamma lines.
            Used to avoid annihilation peak interference. Default is 0.1.
        silent : bool, optional
            If True, suppresses informational messages. Default is True.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the analysis, with columns:
            - 't_c0': Initial counting time (s)
            - 'gamma_energy': Gamma-ray energy (keV)
            - 'isotope': Isotope name
            - 'target': Target element symbol
            - 'foil_number': Foil number
            - 'half_life': Isotope half-life with units
            - 't_c': Array of counting times corresponding to each delay time (s)
            - 't_d': Array of delay times (s)
            
            The DataFrame is sorted by the initial counting time ('t_c0') in ascending order.

        Notes
        -----
        The method uses the Avogadro constant to calculate the number of target atoms,
        and determines the optimal counting time based on the specified detector efficiency.
        Isotopes requiring counting times longer than `t_max` are skipped.
        """
        
        results_list = []
        for _, foil in self.stack.stack.iterrows():
            target = foil['compound']
            if target not in products:
                if not silent:
                    print(f"Element {target} not found in products: {products.keys()}. Skipping.")
                    
                continue

            A = foil['areal_density'] * self.GRAM_TO_KG  # g/cm^2
            for _, isotope in products[target].iterrows():
                iso_name = isotope['Name']
                E   = isotope['E']
                Cs  = isotope['Cs']
                iso = ci.Isotope(iso_name.upper())
                λ   = iso.decay_const()  # decays/s 
                if isinstance(λ, tuple):
                    λ = λ[0]
                    
                M   = iso.mass
                N_T = A * Avogadro / M  # atoms/cm^2
                idx = np.abs(E - foil.mu_E).argmin()  # Find the index of the closest energy to the beam energy 
                σ   = Cs[idx] * self.MILLIBARN_TO_CM2 
                
                A0 = self._A0(σ, N_T, self.proton_flux, self.irradiation_time, λ)  # Bq 
                
                for _, gamma in iso.gammas(I_lim = min_intensity, dE_511 = dE_511).iterrows():
                    E_γ = gamma['energy']  # MeV
                    I_γ = gamma['intensity'] * self.PERCENT_TO_FRACTION  # Convert to fraction
                    ε_γ = self.efficiency_func(E_γ)

                    t_c = self._counting_time(A0, I_γ, t_d, ε_γ, λ)  # s

                    t_c = np.atleast_1d(t_c)
                    if np.isnan(t_c).any():
                        if not silent:
                            print(f"Skipping NaN value for isotope {iso_name} from target {target} with energy {E_γ} MeV.")
                        continue
                    
                    if t_c[0] > t_max:
                        if not silent:
                            print(f"Skipping isotope {iso_name} from target {target} with counting time {t_c[0]} s, which exceeds the maximum allowed time {t_max} s.")
                        continue
                    
                    unit = iso.optimum_units()
                    entry = {'t_c0': t_c[0],
                             'gamma_energy': E_γ,
                             'isotope': iso_name,
                             'target': target,
                             'foil_number': foil['name'][-2:],
                             'half_life': f'{iso.half_life(unit):.2g} {unit}',
                             't_c': t_c,
                             't_d': t_d}
                    results_list.append(entry)
                    
        results_df = pd.DataFrame(results_list).sort_values(by='t_c0', ascending=True, ignore_index=True)
        
        return results_df
    
    def plot(self, 
             counting_times: pd.DataFrame, 
             target: str | None = None, 
             n_gammas: int = 5, 
             title: str = '') -> Figure:
        """
        Plots counting times as a function of delay times for the top gamma lines.

        Parameters
        ----------
        counting_times : pd.DataFrame
            DataFrame containing counting time data, typically the output of the `analyze` method. Needs to have columns 't_d', 't_c', 'gamma_energy', 'isotope', 'target', and 'foil_number'.
        target : str or None, optional
            If provided, only plot data for this target material. Default is None (plot all targets).
        n_gammas : int, optional
            Number of gamma lines to plot, sorted by initial counting time. Default is 5.
        title : str, optional
            Title for the plot. Default is an empty string.

        Returns
        -------
        Figure
            Matplotlib Figure object containing the plot.
        """
        if target is not None:
            counting_times = counting_times[counting_times['target'] == target]
        
        counting_times = counting_times.sort_values(by='t_c0', ascending=True, ignore_index=True)

        plot_row = lambda row: plt.plot(row['t_d'], row['t_c'], label=fr"E: {row['gamma_energy']} keV, iso: $^{{{row['isotope'][:-2]}}}${row['isotope'][-2:]}, target: {row['target']}, foil: {row['foil_number']}")
                                        
        counting_times.head(n_gammas).apply(plot_row, axis=1)
        
        plt.xlabel('Delay Time (s)')
        plt.ylabel('Counting Time (s)')
        plt.title(title)
        
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        return plt.gcf() 
    
    
    def _A0(self, 
            σ:     float | NDArray[float64],  
            N_T:   int   | NDArray[int64],
            Φ:     float | NDArray[float64],
            t_irr: float | NDArray[float64], 
            λ:     float | NDArray[float64]) -> float | NDArray[float64]:
        """
        Calculates the activity A0 from the cross-section σ, target atom density N_T, flux Φ, irradiation time t_irr, and decay constant λ. `t_irr` and `λ` can be either a float or a numpy array. If both are arrays, they must have the same shape.
        
        Parameters
        ----------
        σ: float | NDArray[float64]
            Cross-section in barns (b).
        N_T: int | NDArray[int64]
            Target atom density in atoms/cm^2.
        Φ: float | NDArray[float64]
            Flux in protons/cm^2/s.
        t_irr: float | NDArray[float64]
            Irradiation time in seconds.
        λ: float | NDArray[float64]
            Decay constant in decays/s.
            
        Returns
        -------
        float | NDArray[float64]
            Activity A0 in Bq.
        """
        if isinstance(t_irr, np.ndarray) and isinstance(λ, np.ndarray) and t_irr.shape != λ.shape:
            raise ValueError(f"Shape mismatch: t_irr{t_irr.shape} vs λ{λ.shape}")
        
        A0 = σ * N_T * Φ * (1 - np.exp(-λ * t_irr))
        
        return A0
    
    def _counting_time(self, 
                       A0:  float | NDArray[float64], 
                       I_γ: float | NDArray[float64], 
                       t_d: float | NDArray[float64|int64], 
                       ε_γ: float | NDArray[float64], 
                       λ:   float | NDArray[float64], 
                       N_c: int   | NDArray[int64] = 10_000) -> float | NDArray[float64]:
        """
        Calculates the counting time t_c to count N_c decays, as a function of delay time t_d. All arrays passed for `A0`, `I_γ`, `t_d`, `ε_γ`, `λ` or `N_c` must all have the same shape. 
        
        Parameters
        ----------
        A0: float | NDArray[float]
            Initial activity in Bq.
        I_γ: float | NDArray[float]
            Intensity of the gamma line.
        t_d: np | NDArray[np]
            Delay time in seconds.
        ε_γ: float | NDArray[float]
            Energy of the gamma line in MeV.
        λ: float | NDArray[float]
            Decay constant in decays/s.
        N_c: int | NDArray[int], optional
            Number of counts to achieve. Defaults to 10'000.
            
        Returns
        -------
        float | NDArray[float]
            Counting time in seconds.
        """
        t_c =  np.log(1 - (N_c*λ)/(A0 * I_γ * ε_γ * np.exp(-λ * t_d))) / -λ
        return t_c