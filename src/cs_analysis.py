import pyperclip
import curie as ci
import numpy as np
import pandas as pd
from path import Path
from tendl import Tendl
from numpy import float64
import periodictable as pt
from typing import Literal
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from collections.abc import Iterable
from matplotlib.figure import Figure


class CrossSectionAnalysis:
    """
    This class provides methods to generate, filter, save, load, and plot nuclear cross-section data for isotopes produced by particle beams on target materials. It supports filtering by half-life and cross-section thresholds, and can handle both natural abundance targets and specific isotope lists.
    
    Attributes
    ----------
    target : str | Iterable[str]
        The target material (element symbol or list of isotope strings).
    particle_beam : Literal['proton', 'neutron', 'deuteron']
        The type of particle beam used for activation.
    n_alpha : int
        The span of decays to consider for product generation.
    max_half_life : float
        Maximum half-life (in years) for observed isotopes.
    min_half_life : float
        Minimum half-life (in seconds) for observed isotopes.
    grayzone_half_life : float
        Half-life (in minutes) for isotopes that might be interesting.
    products : tuple[str]
        All product isotopes generated from the target and beam.
    observed_isotopes : list[str]
        Isotopes with half-lives suitable for observation.
    grayzone_isotopes : list[str]
        Isotopes with half-lives in the "gray zone" (potentially observable).
    tendl : Tendl
        Tendl object for cross-section data access.
    loaded_data : pd.DataFrame
        DataFrame containing loaded cross-section data (set by `load_tendl_data`).
    
    Methods
    -------
    `__init__`(target: str | Iterable[str]) -> None:
        Initializes the CrossSectionAnalysis class with target material and particle beam.
    `isotope_overview`(isotopes: Iterable[str] | None = None, max_half_life: float | None = None, min_half_life: float | None = None, grayzone_half_life: float | None = None, copy_to_clipboard: bool = False, print_markdown: bool = False) -> pd.DataFrame:
        Creates an overview DataFrame of isotopes with half-lives and observation status.
    `save_tendl_data`(path: Path, isotopes: Iterable[str] | None = None, Elimit: float = 60, silent: bool = False) -> None:
        Saves Tendl cross-section data for specified isotopes to .npy files.
    `load_tendl_data`(path: Path, isotopes: Iterable[str] | None = None, store: bool = True) -> pd.DataFrame:
        Loads Tendl cross-section data from .npy files for given isotopes.
    `plot_Cs`(title: str, isotopes: pd.DataFrame, low_Cs_threshold: float = 10) -> Figure:
        Plots cross-section data for isotopes, highlighting those above a threshold.
    `filter_products_Cs`(isotopes: Iterable[str] | None = None, Cs_threshold: float = 1e-2, E_limit: float | None = 60, E_beam: float | None = None) -> pd.DataFrame:
        Filters isotopes based on cross-section data and thresholds.
    `_get_isotope`(Z: int, N: int) -> ci.Isotope
        Returns the isotope object for given atomic and neutron numbers.
    `_get_products`(target: str | Iterable[str], n_alpha: int = 3) -> tuple[str]:
        Generates all product isotopes from a target and particle beam.
    `_filter_products_halflife`(isotopes: Iterable[str] | None = None, max_half_life = None, min_half_life = None, grayzone_half_life = None) -> tuple[list[str], list[str]]:
        Filters isotopes based on half-life criteria.
    `_format_dataframe_with_checkboxes`(df: pd.DataFrame) -> pd.DataFrame:
        Formats a DataFrame to display checkboxes for boolean columns.
    `_target_abundance`(target: str | Iterable[str] | None = None) -> dict[str, float]:
        Returns a dictionary of isotope abundances for the target.
    `_load_npy`(path: Path) -> tuple[NDArray[float64], NDArray[float64]]:
        Loads energy and cross-section arrays from a .npy file.
    """
    def __init__(self, 
                 target: str | Iterable[str], 
                 particle_beam: Literal['proton', 'neutron', 'deuteron'] = 'proton', 
                 n_alpha: int = 3, 
                 max_half_life: float = 100, 
                 min_half_life: float = 60, 
                 grayzone_half_life: float = 10) -> None:
        """
        Initializes the CrossSectionAnalysis class.

        Parameters
        ----------
        target : str|Iterable[str]
            The target material, e.g., 'Ag'/'AG', 'Au'/'AU', 'Cu'/'CU'. All naturally occurring isotopes will be used. If a list is provided, it should contain isotopes in the form of strings like '69GA', '71GA'.
        particle_beam : Literal['proton', 'neutron', 'deuteron'], optional
            The type of particle beam used, e.g., 'proton', 'neutron' or 'deuteron'. Defaults to 'proton'.
        n_alpha : int, optional
            The span of decays to consider. Defaults to 3.
        max_half_life : float, optional
            Maximum half-life in years. Defaults to 100.
        min_half_life : float, optional
            Minimum half-life in seconds. Defaults to 60.
        grayzone_half_life : float, optional
            Half-life in minutes to consider isotopes that might be interesting. Defaults to 10.
        """
        
        self.target = target
        self.particle_beam = particle_beam
        self.n_alpha = n_alpha
        
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.grayzone_half_life = grayzone_half_life
        
        # Get all products from the target material and particle beam
        self.products = self._get_products(target, particle_beam, n_alpha)
        
        # Filter products based on half-life criteria
        self.observed_isotopes, self.grayzone_isotopes = self._filter_products_halflife(self.products, max_half_life=max_half_life, min_half_life=min_half_life, grayzone_half_life=grayzone_half_life
        )
        
        # Create Tendl objects for cross-section data
        self.tendl = Tendl(self._target_abundance(target), self.particle_beam)
        
        
    # TODO: Maybe make a single row for "observed", then add "✔", "✘" or "~" for each possible state
    def isotope_overview(self, 
                         isotopes: Iterable[str] | None = None, 
                         max_half_life: float | None = None, 
                         min_half_life: float | None = None, 
                         grayzone_half_life: float | None = None, 
                         copy_to_clipboard: bool = False, 
                         print_markdown: bool = False) -> pd.DataFrame:
        """
        Creates an overview of isotopes with their half-lives and observation status.

        Parameters
        ----------
        isotopes : Iterable[str], optional
            A list of isotope names to include in the overview. If `None`, uses all products found from the target and particle beam upon initialization.
        max_half_life : float, optional
            Maximum half-life in years. Defaults to the class attribute `max_half_life`.
        min_half_life : float, optional
            Minimum half-life in seconds. Defaults to the class attribute `min_half_life`.
        grayzone_half_life : float, optional
            Half-life in minutes to consider isotopes that might be interesting. Defaults to the class attribute `grayzone_half_life`.
        copy_to_clipboard : bool, optional
            If `True`, copies a markdown table created from the dataframe to the clipboard. Defaults to `False`.  
        print_markdown : bool, optional
            If `True`, prints the markdown table to the console. Defaults to `False`.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing isotope information with columns for half-life and observation status.
        """
        if isotopes is None:
            isotopes = self.products
        else:
            isotopes = [iso.upper() for iso in isotopes]  # Ensure isotopes are in uppercase
            
        if max_half_life is None:
            max_half_life = self.max_half_life
            
        if min_half_life is None:
            min_half_life = self.min_half_life
            
        if grayzone_half_life is None:
            grayzone_half_life = self.grayzone_half_life
        
        # Temporarily convert names to Isotope objects for filtering
        isotopes_ci = [ci.Isotope(name.upper()) for name in isotopes]
        isotopes_ci = sorted(isotopes_ci, key=lambda iso: (iso.Z, iso.N), reverse=True) # Sort by Z and N in descending order
        isotopes = [iso._short_name for iso in isotopes_ci]
        
        # If no new filters are applied, use the already filtered lists to avoid unnecessary printing and processing
        if max_half_life == self.max_half_life and min_half_life == self.min_half_life and grayzone_half_life == self.grayzone_half_life:
            observed_isotopes, grayzone_isotopes = self.observed_isotopes, self.grayzone_isotopes
        else:
            observed_isotopes, grayzone_isotopes = self._filter_products_halflife(isotopes, max_half_life=max_half_life, min_half_life=min_half_life, grayzone_half_life=grayzone_half_life)
        
        data = []
        for iso_name in isotopes:
            if iso_name in observed_isotopes:
                observed = True
                not_observed = False
                maybe_observed = False
            elif iso_name in grayzone_isotopes:
                observed = False
                not_observed = False
                maybe_observed = True
            else:
                observed = False
                not_observed = True
                maybe_observed = False
            
            iso = ci.Isotope(iso_name.upper())
            unit = iso.optimum_units()
            hf = iso.half_life(units=unit)
            hf_formatted = f'{hf:.2g} {unit}' if hf != np.inf else 'Stable'
            
            data.append({
                "Isotope": iso_name.title(),
                "Half-life": hf_formatted,
                "Observed": observed,
                "Not Observed": not_observed,
                "Maybe Observed": maybe_observed
            })
        
        df = pd.DataFrame(data)
        formatted_df = self._format_dataframe_with_checkboxes(df)
        
        if copy_to_clipboard:
            markdown_table = formatted_df.to_markdown(index=False)
            pyperclip.copy(markdown_table)
            print("\nMarkdown table copied to clipboard.")
        if print_markdown:
            markdown_table = formatted_df.to_markdown(index=False)
            print(markdown_table)
        
        return df
    
    def save_tendl_data(self, 
                        path: Path, 
                        isotopes: Iterable[str] | None = None, 
                        Elimit: float = 60, 
                        silent: bool = False) -> None:
        """
        Saves Tendl cross-section data for specified isotopes to a .npy file.
        
        Parameters
        ----------
        path : Path
            The path where the data will be saved.
        isotopes : Iterable[str] | None, optional
            An iterable of isotope names to save. e.g., ['Ag108', 'Ag110']. If `None`, saves all products found from the target and particle beam upon initialization.
        Elimit : float, optional
            The energy limit for the cross-section data. Defaults to 60 MeV.
        silent : bool, optional
            If `True`, suppresses print statements. Defaults to `False`.
        """
        if isotopes is None:
            isotopes = self.products
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        for iso_name in isotopes:
            iso = ci.Isotope(iso_name.upper())
            Z, A = str(iso.Z), str(iso.A)
            E, Cs = self.tendl.tendl_data(Z, A, Elimit=Elimit)
            if E is None or Cs is None:
                if not silent:
                    print(f"No cross-section data found for {iso_name}. Skipping.")
                continue
            else:
                np.save(path / f'{iso_name.title()}.npy', np.array([E, Cs]))
    
    def load_tendl_data(self, 
                        path: Path, 
                        isotopes: Iterable[str] | None = None, 
                        store: bool = True) -> pd.DataFrame:
        """
        Loads Tendl cross-section data from .npy files in the specified directory for given isotopes.
        
        Parameters
        ----------
        path : Path
            The path where the data is stored.
        isotopes : list[str], optional
            A list of isotope names to load. e.g., ['Ag108', 'Ag110']. If `None`, loads all products found from the target and particle beam upon initialization.
        store : bool, optional
            If `True`, stores the loaded data in the `loaded_data` attribute of the class. Defaults to `True`.
            
        Returns
        -------
        pd.DataFrame
            A DataFrame containing isotope 'Name', 'E' (energies), and 'Cs' (cross-sections).
        """
        if isotopes is None:
            isotopes = self.products
        
        if not path.exists():
            raise FileNotFoundError(f"The directory {path} does not exist.")
        
        data = {iso.upper(): self._load_npy(path / f'{iso.title()}.npy') for iso in isotopes}
        
        data = pd.DataFrame({
            'Name': [iso.title() for iso in data.keys()],
            'E': [data[iso][0] for iso in data.keys()],
            'Cs': [data[iso][1] for iso in data.keys()]
        })
        
        if store:
            self.loaded_data = data
        return data
    
    def plot_Cs(self, 
                title: str, 
                isotopes: pd.DataFrame, 
                low_Cs_threshold: float = 10) -> Figure:
        """
        Plots cross-section data for isotopes with cross-sections above a specified threshold.
        
        Parameters
        ----------
        title : str
            The title of the plot.
        isotopes : pd.DataFrame
            A DataFrame containing isotope names, energies, and cross-sections. It should have columns 'Name', 'E', and 'Cs'.
        low_Cs_threshold : float, optional
            The threshold for a cross-section to be considered low. Defaults to 10.
        """
        
        
        # Plot the filtered isotopes
        colors = plt.cm.tab20(np.linspace(0, 1, len(isotopes))) # type: ignore
        linestyles = ['-', '--', '-.']
        for i, (_, iso) in enumerate(isotopes.iterrows()):
            iso_name = iso['Name']
            E: NDArray = iso['E']
            Cs: NDArray = iso['Cs']
            
            max_Cs = Cs.max()
            if max_Cs > low_Cs_threshold:
                label = f"{ci.Isotope(iso_name.upper()).TeX}"
                linestyle = linestyles[i % len(linestyles)]
            else:
                label = fr"{ci.Isotope(iso_name.upper()).TeX}$^{'*'}$"
                linestyle = ':'
            
            plt.plot(E, Cs, label=label, color=colors[i % len(colors)], linestyle=linestyle)
            
        
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Cross-section (mb)')
        plt.suptitle(title, y=0.95)
        plt.title(f'Isotopes marked with * have a maximum cross-section below {low_Cs_threshold} mb.', fontsize=10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        
        return plt.gcf()  # Return the current figure for further manipulation or saving
    
    def filter_products_Cs(self, 
                            isotopes: Iterable[str] | None = None, 
                            Cs_threshold: float = 1e-2, 
                            E_limit: float | None = 60, 
                            E_beam: float | None = None) -> pd.DataFrame:
        """
        Filters isotopes based on cross-section data above a specified threshold.
        
        Parameters
        ----------
        isotopes : Iterable[str] | None, optional
            An Iterable of isotope names to filter. If `None`, uses the observed isotopes from the loaded data from `load_tendl_data`. Defaults to `None`.
        Cs_threshold : float, optional
            The threshold for cross-sections to be considered. Defaults to 1e-2 mb.
        E_limit : float | None, optional
            The upper energy limit for the cross-section data. If `None`, uses the entire range from tendl. Defaults to 60 MeV.
        E_beam : float | None, optional
            The energy of the particle beam. If provided, will be the energy which is used to filter the cross-section data. If `None`, searches the entire energy range specified by `E_limit` for a cross-section above `Cs_threshold`. Defaults to `None`.
        """
        
        data = {}
        if isotopes is None:
            if not hasattr(self, 'loaded_data'):
                raise ValueError("No loaded data found. Please load data using `load_tendl_data()` method before plotting or provide isotopes.")   
            
            for _, iso in self.loaded_data.iterrows():
                E = iso['E']
                Cs = iso['Cs']
                iso_name = iso['Name'].upper() # type: ignore
                if iso_name in self.observed_isotopes:
                    data[iso_name] = (E, Cs)
                    
        
        # If isotopes is a list of strings
        elif all(isinstance(iso, str) for iso in isotopes):
            # Use loaded data if available
            if hasattr(self, 'loaded_data'):
                for iso_name in isotopes:
                    iso_name = iso_name.title()
                    if iso_name in self.loaded_data:
                        E, Cs = self.loaded_data[iso_name]
                        data[iso_name] = (E, Cs)
                    else:
                        iso = ci.Isotope(iso_name.upper())
                        Z, A = str(iso.Z), str(iso.A)
                        E, Cs = self.tendl.tendl_data(Z, A, Elimit=E_limit)
                        if E is not None and Cs is not None:
                            data[iso._short_name] = (E, Cs)
            
            # If no loaded data, fetch from Tendl
            else:            
                for iso in isotopes:
                    iso = ci.Isotope(iso.upper())
                    Z, A = str(iso.Z), str(iso.A)
                    E, Cs = self.tendl.tendl_data(Z, A, Elimit=E_limit)
                    if E is not None and Cs is not None:
                        data[iso._short_name] = (E, Cs)

        else:
            raise TypeError(f"Invalid type for isotopes: {type(isotopes)}. Expected list of strings, or None to use loaded data for all products.")
        
        if E_beam is None:
            conditon = lambda E, Cs: np.max(Cs) > Cs_threshold
        else:
            conditon = lambda E, Cs: Cs[np.abs(E - E_beam).argmin()] > Cs_threshold
        
        # Filter isotopes based on cross-section threshold
        filtered_isotopes = {'Name': [], 'E': [], 'Cs': []}
        for i, (iso_name, (E, Cs)) in enumerate(data.items()):
            if conditon(E, Cs):
                filtered_isotopes['Name'].append(iso_name.title())
                filtered_isotopes['E'].append(E)
                filtered_isotopes['Cs'].append(Cs)
        
        
        # Sort isotopes by maximum cross-section value in descending order
        sorted_filtered_isotopes = pd.DataFrame(filtered_isotopes).sort_values(by='Cs', key=lambda x: x.apply(np.max), ascending=False, ignore_index=True)
        
        print(f'Found {len(sorted_filtered_isotopes)}/{len(data)} isotopes with cross-sections above {Cs_threshold} mb' + f'{" at E = " + str(E_beam) + " MeV:" if E_beam is not None else "" + ":"}')
        print(' '.join(sorted_filtered_isotopes['Name'].tolist()))
        
        return sorted_filtered_isotopes 
        
    def _get_isotope(self, Z: int, N: int) -> ci.Isotope:
        """
        Returns the isotope with atomic number Z and neutron number N.
        
        Parameters
        ----------
        Z : int
            Atomic number of the isotope.
        N : int
            Neutron number of the isotope.
            
        Returns
        -------
        ci.Isotope
            The isotope object corresponding to the given Z and N.
        """
        A = Z + N
        element = pt.elements[Z].symbol.upper() 
        iso_name = f'{A}{element}'
        return ci.Isotope(iso_name)
        
    def _get_products(self, 
                      target: str | Iterable[str], 
                      particle_beam: Literal['proton', 'neutron', 'deuteron'] = 'proton', 
                      n_alpha: int = 3) -> tuple[str]:
        """
        Creates a list of all isotopes produced by a particle beam on a target material. Material is assumed to be in natural abundance.
        
        Parameters
        ----------
        target : str | Iterable[str]
            The target material, e.g., 'Ag'/'AG', 'Au'/'AU', 'Cu'/'CU'. All naturally occurring isotopes will be used. If a list is provided, it should contain isotopes in the form of strings like '69GA', '71GA'.
        particle_beam : Literal['proton', 'neutron', 'deuteron'], optional
            The type of particle beam used, e.g., 'proton', 'neutron' or 'deuteron'. Defaults to 'proton'.
        n_alpha : int, optional
            The span of decays to consider. Defaults to 3.
            
        Returns
        -------
        tuple[ci.Isotope]
            A tuple of isotopes produced by the particle beam on the target material.
            
        Raises
        ------
        ValueError
            If an unsupported particle beam type is provided.
        TypeError
            If the target is not a string or a list of strings.
        """
        
        if particle_beam == 'proton':
            Z_add, N_add = 1, 0  # Proton beam adds 1 proton
        elif particle_beam == 'neutron':
            Z_add, N_add = 0, 1  # Neutron beam adds 1 neutron
        elif particle_beam == 'deuteron':
            Z_add, N_add = 1, 1
        else:
            raise ValueError(f"Unsupported particle beam type: {particle_beam}. Use 'proton', 'neutron', or 'deuteron'.")   
        
        isotopes: list[ci.Isotope]
        if isinstance(target, str):
            element = ci.Element(target)
            isotopes = [ci.Isotope(iso.upper()) for iso in element.abundances['isotope']]   
            
        elif isinstance(target, Iterable):
            isotopes: list[ci.Isotope] = [ci.Isotope(iso.upper()) for iso in target]
            
        else:
            raise TypeError(f"Target must be a string representing an element or an iterable of isotope strings, got {type(target)}.")
        
        isotopes = sorted(isotopes, key=lambda iso: iso.A, reverse=True) # Sort by mass number A in descending order
        
        products = []
        product_names = set()  # To avoid duplicates
        for iso in isotopes:
            Z, N = iso.Z, iso.N
            Z_prod = Z + Z_add
            N_prod = N + N_add
            direct_product = self._get_isotope(Z_prod, N_prod)
            if direct_product._short_name not in product_names:
                products.append(direct_product)
                product_names.add(direct_product._short_name)
            # Create new product isotopes by adding protons or neutrons
            for Z in range(Z_prod, Z_prod-2*n_alpha -1, -1):
                for N in range(N_prod, N_prod-2*n_alpha -1, -1):
                    product = self._get_isotope(Z, N)
                    if product._short_name not in product_names:
                        products.append(product)
                        product_names.add(product._short_name)
                        
        return tuple(product_names)
          
    def _filter_products_halflife(self, 
                                  isotopes: Iterable[str] | None = None, 
                                  max_half_life = None, 
                                  min_half_life = None, 
                                  grayzone_half_life = None) -> tuple[list[str], list[str]]:
        """
        Filters isotopes based on their half-life. Adds the results to the class attributes `observed_isotopes` and `grayzone_isotopes`.

        Parameters
        ----------
        isotopes : list[str], optional
            A list of isotope names to filter. If `None`, uses all products found from the target and particle beam upon initialization.
        max_half_life : float, optional
            Maximum half-life in years. Defaults to the class attribute `max_half_life`.
        min_half_life : float, optional
            Minimum half-life in seconds. Defaults to the class attribute `min_half_life`.
        grayzone_half_life : float, optional
            Half-life in minutes to consider isotopes that might be interesting. Defaults to the class attribute `grayzone_half_life`.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing two lists:
            - `observed_isotopes`: Isotopes with a half-life between `grayzone_half_life` and `max_half_life`.
            - `potentially_observed_isotopes`: Isotopes with a half-life between `min_half_life` and `grayzone_half_life`.
        """
        
        if isotopes is None:
            isotopes = self.products
            
        if max_half_life is None:
            max_half_life = self.max_half_life
            
        if min_half_life is None:
            min_half_life = self.min_half_life
            
        if grayzone_half_life is None:
            grayzone_half_life = self.grayzone_half_life
            
        observed_isotopes = []
        potentially_observed_isotopes = []
        
        for iso_name in isotopes:
            iso = ci.Isotope(iso_name.upper())
            λ = iso.half_life()
            if isinstance(λ, tuple):
                λ = λ[0]  
                 
            if grayzone_half_life*60 < λ < max_half_life*365*24*3600:
                observed_isotopes.append(iso._short_name)
            elif min_half_life < λ < grayzone_half_life*60:
                potentially_observed_isotopes.append(iso._short_name)
        
        self.observed_isotopes = observed_isotopes
        self.grayzone_isotopes = potentially_observed_isotopes
        
        n_isotopes = len(list(isotopes))
        print(f"Found {len(observed_isotopes)}/{n_isotopes} isotopes to be observed based on half-life:")
        print(' '.join(list(map(str.title, observed_isotopes))))
        
        print()
        
        print(f"Found {len(potentially_observed_isotopes)}/{n_isotopes} isotopes which might be observed based on half-life.")
        print(' '.join(list(map(str.title, potentially_observed_isotopes))))
        
        return observed_isotopes, potentially_observed_isotopes

    def _format_dataframe_with_checkboxes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats a DataFrame to replace boolean values with checkboxes.
    
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to format.
    
        Returns
        -------
        pd.DataFrame
            A formatted DataFrame with checkboxes instead of boolean values.
        """
        formatted_df = df.copy()
        for col in ["Observed", "Not Observed", "Maybe Observed"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: "✔" if x else "✘")
        return formatted_df
    
    def _target_abundance(self, 
                          target: str | Iterable[str] | None = None) -> dict[str, float]:
        """
        Returns the natural abundance of isotopes in the target material.
        
        Parameters
        ----------
        target : str|Iterable[str]
            The target material, e.g., 'Ag'/'AG', 'Au'/'AU', 'Cu'/'CU'. If an iterable is provided, it should contain isotopes in the form of strings like '69GA', '71GA'.
            
        Returns
        -------
        dict[str, float]
            A dictionary with isotope names as keys and their natural abundances as values.
        """
        if target is None:
            target = self.target
        
        if isinstance(target, str):
            element = ci.Element(target)
            abundances = {name: abundance/100 for name, abundance in zip(element.abundances['isotope'], element.abundances['abundance'])}
        
        elif isinstance(target, Iterable):
            abundances = {iso._short_name: iso.abundance/100 for iso in [ci.Isotope(iso.upper()) for iso in target]}
            
        else:
            raise TypeError(f"Target must be a string representing an element or a list of isotope strings, got {type(target)}.")
        
        return abundances 

    def _load_npy(self, 
                  path: Path) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Loads cross-section data from a .npy file.
        
        Parameters
        ----------
        path : Path
            The path to the .npy file.
            
        Returns
        -------
        tuple[NDArray[float64], NDArray[float64]]
            A tuple containing two numpy arrays: energies and cross-sections.
        """
        if not path.exists():
            print(f"File {path} does not exist. Returning empty arrays.")
            return np.array([], dtype=float), np.array([], dtype=float)
        
        E, Cs = np.load(path)
        return E, Cs     