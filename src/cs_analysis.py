import pyperclip
import curie as ci
import numpy as np
import pandas as pd
from path import Path
from tendl import Tendl
from numpy import float64
import periodictable as pt
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Literal, NewType
from collections.abc import Iterable
from matplotlib.figure import Figure

# Custom types for nuclear physics data
IsotopeName = NewType('IsotopeName', str)  # e.g., "108AG", "63CU"
HalfLife = NewType('HalfLife', str)        # e.g., "2.4 m", "25 s", "Stable"
Energy = NewType('Energy', float)          # Energy in MeV
CrossSection = NewType('CrossSection', float)  # Cross-section in mb
ObservationStatus = NewType('ObservationStatus', str)  # "✔", "✘", "~"


class CrossSectionAnalysis:
    """
    Analyzes nuclear cross-section data for isotopes produced by particle beam activation.

    This class generates, filters, saves, loads, and plots nuclear cross-section data
    for isotopes produced by particle beams on target materials. It supports filtering
    by half-life and cross-section thresholds, and handles both natural abundance
    targets and specific isotope lists.

    Attributes
    ----------
    target : str | Iterable[str]
        Target material (element symbol or list of isotope strings).
    particle_beam : Literal['proton', 'neutron', 'deuteron']
        Type of particle beam used for activation.
    n_alpha : int
        Span of decays to consider for product generation.
    max_half_life : float
        Maximum half-life (years) for observed isotopes.
    min_half_life : float
        Minimum half-life (seconds) for observed isotopes.
    grayzone_half_life : float
        Half-life (minutes) for potentially interesting isotopes.
    products : tuple[str, ...]
        All product isotopes generated from target and beam.
    observed_isotopes : list[str]
        Isotopes with half-lives suitable for observation.
    grayzone_isotopes : list[str]
        Isotopes with half-lives in the "gray zone".
    tendl : Tendl
        Tendl object for cross-section data access.
    loaded_data : pd.DataFrame, optional
        DataFrame containing loaded cross-section data.

    Methods
    -------
    `isotope_overview`(isotopes=None, max_half_life=None, min_half_life=None, grayzone_half_life=None, copy_to_clipboard=False, print_markdown=False) -> pd.DataFrame
        Creates overview DataFrame of isotopes with half-lives and observation status.
    `save_tendl_data`(path, isotopes=None, Elimit=60, silent=False) -> None
        Saves Tendl cross-section data for specified isotopes to .npy files.
    `load_tendl_data`(path, isotopes=None, store=True) -> pd.DataFrame
        Loads Tendl cross-section data from .npy files.
    `plot_Cs`(title, isotopes, low_Cs_threshold=10) -> Figure
        Plots cross-section data, highlighting isotopes above threshold.
    `filter_products_Cs`(isotopes=None, Cs_threshold=1e-2, E_limit=60, E_beam=None) -> pd.DataFrame
        Filters isotopes based on cross-section data and thresholds.
    """

    # Class constants
    SECONDS_PER_YEAR = 365 * 24 * 3600
    SECONDS_PER_MINUTE = 60
    DEFAULT_ENERGY_LIMIT: Energy = Energy(60.0)  # MeV
    DEFAULT_CS_THRESHOLD: CrossSection = CrossSection(1e-2)  # mb

    target: str | Iterable[str]
    particle_beam: Literal["proton", "neutron", "deuteron"]
    n_alpha: int
    max_half_life: float
    min_half_life: float
    grayzone_half_life: float
    products: tuple[str, ...]
    observed_isotopes: list[str]
    grayzone_isotopes: list[str]
    tendl: Tendl
    loaded_data: pd.DataFrame | None = None

    def __init__(
        self,
        target: str | Iterable[str],
        particle_beam: Literal["proton", "neutron", "deuteron"] = "proton",
        n_alpha: int = 3,
        max_half_life: float = 100,
        min_half_life: float = 60,
        grayzone_half_life: float = 10,
    ) -> None:
        """
        Initialize the CrossSectionAnalysis class.

        Parameters
        ----------
        target : str | Iterable[str]
            Target material (e.g., 'Ag', 'Au', 'Cu') or list of isotopes.
        particle_beam : Literal['proton', 'neutron', 'deuteron'], optional
            Type of particle beam, by default 'proton'.
        n_alpha : int, optional
            Span of decays to consider, by default 3.
        max_half_life : float, optional
            Maximum half-life in years, by default 100.
        min_half_life : float, optional
            Minimum half-life in seconds, by default 60.
        grayzone_half_life : float, optional
            Half-life in minutes for gray zone isotopes, by default 10.
        """

        self.target = target
        self.particle_beam = particle_beam
        self.n_alpha = n_alpha
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.grayzone_half_life = grayzone_half_life

        self.products = self._get_products(target, particle_beam, n_alpha)
        self.observed_isotopes, self.grayzone_isotopes = self._filter_products_halflife(
            self.products,
            max_half_life=max_half_life,
            min_half_life=min_half_life,
            grayzone_half_life=grayzone_half_life,
        )
        self.tendl = Tendl(self._target_abundance(target), self.particle_beam)

    def isotope_overview(
        self,
        isotopes: Iterable[IsotopeName] | None = None,
        max_half_life: float | None = None,
        min_half_life: float | None = None,
        grayzone_half_life: float | None = None,
        copy_to_clipboard: bool = False,
        print_markdown: bool = False,
    ) -> pd.DataFrame:
        """
        Create overview of isotopes with half-lives and observation status.

        Parameters
        ----------
        isotopes : Iterable[str] | None, optional
            Isotope names to include. If None, uses all products.
        max_half_life : float | None, optional
            Maximum half-life in years. If None, uses class default.
        min_half_life : float | None, optional
            Minimum half-life in seconds. If None, uses class default.
        grayzone_half_life : float | None, optional
            Half-life in minutes for gray zone. If None, uses class default.
        copy_to_clipboard : bool, optional
            Whether to copy markdown table to clipboard, by default False.
        print_markdown : bool, optional
            Whether to print markdown table, by default False.

        Returns
        -------
        pd.DataFrame
            DataFrame with isotope information and observation status.
        """
        if isotopes is None:
            isotopes = [IsotopeName(iso) for iso in self.products]
        isotopes = [IsotopeName(iso.upper()) for iso in isotopes]

        max_half_life = max_half_life or self.max_half_life
        min_half_life = min_half_life or self.min_half_life
        grayzone_half_life = grayzone_half_life or self.grayzone_half_life

        observed_isotopes, grayzone_isotopes = self._filter_products_halflife(
            [str(iso) for iso in isotopes],
            max_half_life=max_half_life,
            min_half_life=min_half_life,
            grayzone_half_life=grayzone_half_life,
            silent=True,  # Suppress duplicate printing
        )

        # Create and sort isotope objects
        isotope_objects = [ci.Isotope(name) for name in isotopes]
        isotope_objects.sort(
            key=lambda iso: (iso.Z, iso.N), reverse=True
        )  # Sort by Z and N in descending order

        data = []
        for iso in isotope_objects:
            iso_name = IsotopeName(iso._short_name)
            status = self._get_isotope_status_symbol(
                iso_name, [IsotopeName(iso) for iso in observed_isotopes], [IsotopeName(iso) for iso in grayzone_isotopes]
            )

            unit = iso.optimum_units()
            hf = iso.half_life(units=unit)
            hf_formatted = f"{hf:.2g} {unit}" if hf != np.inf else "Stable"

            data.append(
                {
                    "Isotope": iso_name.title(),
                    "Half-life": hf_formatted,
                    "Status": status,
                }
            )

        df = pd.DataFrame(data)

        if copy_to_clipboard:
            markdown_table = df.to_markdown(index=False)
            pyperclip.copy(markdown_table)
            print("\nMarkdown table copied to clipboard.")
        if print_markdown:
            markdown_table = df.to_markdown(index=False)
            print(markdown_table)

        return df

    def save_tendl_data(
        self,
        path: Path,
        isotopes: Iterable[IsotopeName] | None = None,
        Elimit: Energy = DEFAULT_ENERGY_LIMIT,
        silent: bool = False,
    ) -> None:
        """
        Save Tendl cross-section data for specified isotopes to .npy files.

        Parameters
        ----------
        path : Path
            Directory path where data will be saved.
        isotopes : Iterable[str] | None, optional
            Isotope names to save. If None, saves all products.
        Elimit : float, optional
            Energy limit for cross-section data in MeV, by default 60.
        silent : bool, optional
            Whether to suppress print statements, by default False.

        Raises
        ------
        FileNotFoundError
            If the path doesn't exist and can't be created.
        """
        if isotopes is None:
            isotopes = [IsotopeName(iso) for iso in self.products]

        path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        for iso_name in isotopes:
            iso = ci.Isotope(iso_name.upper())
            Z, A = str(iso.Z), str(iso.A)

            try:
                E, Cs = self.tendl.tendl_data(Z, A, Elimit=Elimit)
                if E is None or Cs is None:
                    if not silent:
                        print(f"No cross-section data found for {iso_name}. Skipping.")
                    continue
                np.save(path / f"{iso_name.title()}.npy", np.array([E, Cs]))
            except Exception as e:
                if not silent:
                    print(f"Error processing {iso_name}: {e}")

    def load_tendl_data(
        self, path: Path, isotopes: Iterable[IsotopeName] | None = None, store: bool = True
    ) -> pd.DataFrame:
        """
        Load Tendl cross-section data from .npy files.

        Parameters
        ----------
        path : Path
            Directory path containing the .npy files.
        isotopes : Iterable[str] | None, optional
            Isotope names to load. If None, loads all products.
        store : bool, optional
            Whether to store data in loaded_data attribute, by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Name', 'E' (energies), 'Cs' (cross-sections).

        Raises
        ------
        FileNotFoundError
            If the specified directory doesn't exist.
        """
        if isotopes is None:
            isotopes = [IsotopeName(iso) for iso in self.products]

        if not path.exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")

        loaded_data = {}
        for iso in isotopes:
            try:
                E, Cs = self._load_npy(path / f"{iso.title()}.npy")
                if len(E) > 0 and len(Cs) > 0:
                    loaded_data[iso.upper()] = (E, Cs)
            except FileNotFoundError:
                print(f"Warning: No data file found for {iso}")

        names, energies, cross_sections = zip(
            *[(iso.title(), data[0], data[1]) for iso, data in loaded_data.items()]
        )

        df = pd.DataFrame({"Name": names, "E": energies, "Cs": cross_sections})

        if store:
            self.loaded_data = df
        return df

    def plot_Cs(
        self, title: str, isotopes: pd.DataFrame, low_Cs_threshold: float = 10
    ) -> Figure:
        """
        Plot cross-section data for isotopes.

        Parameters
        ----------
        title : str
            Plot title.
        isotopes : pd.DataFrame
            DataFrame with columns 'Name', 'E', 'Cs'.
        low_Cs_threshold : float, optional
            Threshold for marking isotopes with low cross-sections, by default 10.

        Returns
        -------
        Figure
            Matplotlib Figure object containing the plot.
        """
        colors = plt.cm.tab20(np.linspace(0, 1, len(isotopes)))  # type: ignore
        linestyles = ["-", "--", "-."]

        for i, (_, iso) in enumerate(isotopes.iterrows()):
            iso_name = iso["Name"]
            E: NDArray = iso["E"]
            Cs: NDArray = iso["Cs"]

            max_Cs = Cs.max()
            if max_Cs > low_Cs_threshold:
                label = f"{ci.Isotope(iso_name.upper()).TeX}"
                linestyle = linestyles[i % len(linestyles)]
            else:
                label = f"{ci.Isotope(iso_name.upper()).TeX}$^{{'*'}}$"
                linestyle = ":"

            plt.plot(
                E, Cs, label=label, color=colors[i % len(colors)], linestyle=linestyle
            )

        plt.xlabel("Energy (MeV)")
        plt.ylabel("Cross-section (mb)")
        plt.suptitle(title, y=0.95)
        plt.title(
            f"Isotopes marked with * have max cross-section below {low_Cs_threshold} mb.",
            fontsize=10,
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        return plt.gcf()  # Return the current figure

    def filter_products_Cs(
        self,
        isotopes: Iterable[IsotopeName] | None = None,
        Cs_threshold: CrossSection = DEFAULT_CS_THRESHOLD,
        E_limit: Energy | None = DEFAULT_ENERGY_LIMIT,
        E_beam: Energy | None = None,
    ) -> pd.DataFrame:
        """
        Filter isotopes based on cross-section thresholds.

        Parameters
        ----------
        isotopes : Iterable[str] | None, optional
            Isotope names to filter. If None, uses observed isotopes from loaded data.
        Cs_threshold : float, optional
            Cross-section threshold in mb, by default 1e-2.
        E_limit : float | None, optional
            Upper energy limit in MeV, by default 60.
        E_beam : float | None, optional
            Specific beam energy in MeV. If None, searches entire range.

        Returns
        -------
        pd.DataFrame
            DataFrame with filtered isotopes sorted by maximum cross-section.

        Raises
        ------
        AttributeError
            If no loaded data available and isotopes is None.
        TypeError
            If isotopes parameter has invalid type.
        """
        data = self._collect_isotope_data(isotopes, E_limit)
        filtered_data = self._apply_cs_filter(data, Cs_threshold, E_beam)
        return self._format_filtered_results(filtered_data, Cs_threshold, E_beam)

    def _get_isotope_status_symbol(
        self, iso_name: IsotopeName, observed_isotopes: list[IsotopeName], grayzone_isotopes: list[IsotopeName]
    ) -> ObservationStatus:
        """
        Get single status symbol for isotope observation status.

        Parameters
        ----------
        iso_name : str
            Name of the isotope.
        observed_isotopes : list[str]
            List of observed isotope names.
        grayzone_isotopes : list[str]
            List of grayzone isotope names.

        Returns
        -------
        str
            Status symbol: "✔" (observed), "~" (maybe), "✘" (not observed).
        """
        if iso_name in observed_isotopes:
            return ObservationStatus("✔")  # Observed
        elif iso_name in grayzone_isotopes:
            return ObservationStatus("~")  # Maybe observed (grayzone)
        else:
            return ObservationStatus("✘")  # Not observed

    def _collect_isotope_data(
        self, isotopes: Iterable[IsotopeName] | None, E_limit: Energy | None
    ) -> dict[IsotopeName, tuple[NDArray[float64], NDArray[float64]]]:
        """
        Collects cross-section data for specified isotopes.

        Parameters
        ----------
        isotopes : Iterable[str] or None
            List of isotope names to collect data for. If None, uses isotopes from loaded data.
        E_limit : float or None
            Optional energy limit to filter the data.

        Returns
        -------
        dict[str, tuple[NDArray[float64], NDArray[float64]]]
            Dictionary mapping isotope names to tuples of (energy array, cross-section array).

        Raises
        ------
        AttributeError
            If no cross-section data is loaded and isotopes is None.
        TypeError
            If isotopes is not an iterable of strings or None.
        """
        data = {}

        if isotopes is None:
            if self.loaded_data is None:
                raise AttributeError(
                    "No cross-section data loaded. "
                    "Call `load_tendl_data()` first or provide isotopes parameter."
                )

            for _, iso in self.loaded_data.iterrows():
                iso_name = IsotopeName(iso["Name"].upper())
                if iso_name in [IsotopeName(obs_iso) for obs_iso in self.observed_isotopes]:
                    data[iso_name] = (iso["E"], iso["Cs"])

        # If isotopes is a list of strings
        elif all(isinstance(iso, str) for iso in isotopes):
            data = self._fetch_isotope_data(isotopes, E_limit)
        else:
            raise TypeError(
                f"Invalid isotope type: {type(isotopes)}. "
                "Expected Iterable[str] or None."
            )

        return data

    def _fetch_isotope_data(
        self, isotopes: Iterable[IsotopeName], E_limit: Energy | None
    ) -> dict[IsotopeName, tuple[NDArray[float64], NDArray[float64]]]:
        """
        Fetches cross-section data for a list of isotopes, either from pre-loaded data or from the TENDL database.

        Parameters
        ----------
        isotopes : Iterable[str]
            An iterable of isotope names (e.g., '197AU', '63CU') for which to fetch cross-section data.
        E_limit : float | None
            Optional upper energy limit for the cross-section data. If None, fetches all available data.

        Returns
        -------
        dict[str, tuple[NDArray[float64], NDArray[float64]]]
            A dictionary mapping isotope names (in uppercase) to a tuple containing:
            - Energies (NDArray[float64])
            - Cross-sections (NDArray[float64])

        Notes
        -----
        - Tries to use pre-loaded data if available; otherwise, fetches from the TENDL database.
        - If data cannot be fetched for an isotope, a warning is printed and the isotope is skipped.
        """
        data = {}

        for iso_name in isotopes:
            # Try loaded data first
            if self.loaded_data is not None:
                matching_rows = self.loaded_data[
                    self.loaded_data["Name"].str.upper() == iso_name.upper()
                ]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    data[IsotopeName(iso_name.upper())] = (row["E"], row["Cs"])
                    continue

            # Fetch from Tendl if not in loaded data
            try:
                iso = ci.Isotope(iso_name.upper())
                E, Cs = self.tendl.tendl_data(str(iso.Z), str(iso.A), Elimit=E_limit)
                if E is not None and Cs is not None:
                    data[IsotopeName(iso._short_name)] = (E, Cs)
            except Exception as e:
                print(f"Warning: Could not fetch data for {iso_name}: {e}")

        return data

    def _apply_cs_filter(
        self,
        data: dict[IsotopeName, tuple[NDArray[float64], NDArray[float64]]],
        Cs_threshold: CrossSection,
        E_beam: Energy | None,
    ) -> dict[IsotopeName, tuple[NDArray[float64], NDArray[float64]]]:
        """
        Filters the input data dictionary based on a cross-section (Cs) threshold.
        For each isotope in the input data, checks if the cross-section exceeds the specified threshold.
        If `E_beam` is provided, the cross-section at the energy closest to `E_beam` is compared to the threshold.
        If `E_beam` is None, the maximum cross-section value is compared to the threshold.

        Parameters
        ----------
        data : dict[str, tuple[NDArray[float64], NDArray[float64]]]
            Dictionary mapping isotope names to tuples of (energy array, cross-section array).
        Cs_threshold : float
            Cross-section threshold for filtering.
        E_beam : float | None
            Beam energy to use for threshold comparison. If None, uses maximum cross-section.

        Returns
        -------
        dict[str, tuple[NDArray[float64], NDArray[float64]]]
            Filtered dictionary containing only isotopes that meet the threshold condition.
        """
        if E_beam is None:
            condition = lambda _, Cs: np.max(Cs) > Cs_threshold
        else:
            condition = lambda E, Cs: Cs[np.abs(E - E_beam).argmin()] > Cs_threshold

        return {
            iso_name: (E, Cs) for iso_name, (E, Cs) in data.items() if condition(E, Cs)
        }

    def _format_filtered_results(
        self,
        filtered_data: dict[IsotopeName, tuple[NDArray[float64], NDArray[float64]]],
        Cs_threshold: CrossSection,
        E_beam: Energy | None,
    ) -> pd.DataFrame:
        """
        Formats and sorts filtered cross-section results into a DataFrame.

        Parameters
        ----------
        filtered_data : dict[str, tuple[NDArray[float64], NDArray[float64]]]
            Dictionary mapping isotope names to tuples of (energy array, cross-section array).
        Cs_threshold : float
            Threshold value for cross-section (mb) used for filtering.
        E_beam : float or None
            Beam energy in MeV, or None if not specified.

        Returns
        -------
        pd.DataFrame
            DataFrame containing isotope names, energies, and cross-sections,
            sorted by maximum cross-section in descending order.

        Prints
        ------
        Summary of the number of isotopes found above the threshold and their names.
        """
        result_data = {
            "Name": [iso_name.title() for iso_name in filtered_data.keys()],
            "E": [E for E, _ in filtered_data.values()],
            "Cs": [Cs for _, Cs in filtered_data.values()],
        }

        df = pd.DataFrame(result_data).sort_values(
            by="Cs", key=lambda x: x.apply(np.max), ascending=False, ignore_index=True
        )

        total_count = len(filtered_data)
        energy_info = f" at E = {E_beam} MeV" if E_beam is not None else ""
        print(
            f"Found {len(df)}/{total_count} isotopes with cross-sections "
            f"above {Cs_threshold} mb{energy_info}:"
        )
        print(" ".join(df["Name"].tolist()))

        return df

    def _get_products(
        self,
        target: str | Iterable[str],
        particle_beam: Literal["proton", "neutron", "deuteron"] = "proton",
        n_alpha: int = 3,
    ) -> tuple[str, ...]:
        """
        Generate all isotopes produced by particle beam on target.

        Parameters
        ----------
        target : str | Iterable[str]
            Target material or list of isotopes.
        particle_beam : Literal['proton', 'neutron', 'deuteron']
            Type of particle beam.
        n_alpha : int
            Span of decays to consider.

        Returns
        -------
        tuple[str, ...]
            Tuple of product isotope names.

        Raises
        ------
        ValueError
            If unsupported particle beam type.
        TypeError
            If target has invalid type.
        """
        beam_changes = {"proton": (1, 0), "neutron": (0, 1), "deuteron": (1, 1)}

        if particle_beam not in beam_changes:
            raise ValueError(
                f"Unsupported particle beam: {particle_beam}. "
                f"Use: {', '.join(beam_changes.keys())}"
            )

        Z_add, N_add = beam_changes[particle_beam]

        if isinstance(target, str):
            element = ci.Element(target)
            isotopes = [
                ci.Isotope(iso.upper()) for iso in element.abundances["isotope"]
            ]
        elif isinstance(target, Iterable):
            isotopes = [ci.Isotope(iso.upper()) for iso in target]
        else:
            raise TypeError(f"Target must be str or Iterable[str], got {type(target)}")

        isotopes.sort(key=lambda iso: iso.A, reverse=True)

        product_names = set()
        for iso in isotopes:
            Z_prod, N_prod = iso.Z + Z_add, iso.N + N_add

            # Add direct product
            direct_product = self._get_isotope(Z_prod, N_prod)
            product_names.add(direct_product._short_name)

            # Add decay products
            for Z in range(Z_prod, Z_prod - 2 * n_alpha - 1, -1):
                for N in range(N_prod, N_prod - 2 * n_alpha - 1, -1):
                    if Z > 0 and N > 0:  # Valid nuclei only
                        product = self._get_isotope(Z, N)
                        product_names.add(product._short_name)

        return tuple(product_names)

    def _filter_products_halflife(
        self,
        isotopes: Iterable[str],
        max_half_life: float,
        min_half_life: float,
        grayzone_half_life: float,
        silent: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Filter isotopes by half-life criteria.

        Parameters
        ----------
        isotopes : Iterable[str]
            Isotope names to filter.
        max_half_life : float
            Maximum half-life in years.
        min_half_life : float
            Minimum half-life in seconds.
        grayzone_half_life : float
            Half-life threshold in minutes for grayzone classification.
        silent : bool, optional
            If True, suppresses print statements, by default False.

        Returns
        -------
        tuple[list[str], list[str]]
            Observed and grayzone isotope lists.
        """
        observed_isotopes = []
        grayzone_isotopes = []

        for iso_name in isotopes:
            try:
                iso = ci.Isotope(iso_name.upper())
                hf = iso.half_life()
                if isinstance(hf, tuple):
                    hf = hf[0]

                if (
                    grayzone_half_life * self.SECONDS_PER_MINUTE
                    < hf
                    < max_half_life * self.SECONDS_PER_YEAR
                ):
                    observed_isotopes.append(iso._short_name)
                elif min_half_life < hf < grayzone_half_life * self.SECONDS_PER_MINUTE:
                    grayzone_isotopes.append(iso._short_name)
            except Exception:
                # Skip isotopes that can't be processed
                continue
        if not silent:
            n_total = len(list(isotopes))
            print(f"Found {len(observed_isotopes)}/{n_total} isotopes for observation:")
            print(" ".join(map(str.title, observed_isotopes)))
            print(f"\nFound {len(grayzone_isotopes)}/{n_total} grayzone isotopes:")
            print(" ".join(map(str.title, grayzone_isotopes)))
            print(f"\n")

        return observed_isotopes, grayzone_isotopes

    def _get_isotope(self, Z: int, N: int) -> ci.Isotope:
        """
        Get isotope object from atomic and neutron numbers.

        Parameters
        ----------
        Z : int
            Atomic number.
        N : int
            Neutron number.

        Returns
        -------
        ci.Isotope
            Isotope object.
        """
        A = Z + N
        element = pt.elements[Z].symbol.upper()
        return ci.Isotope(f"{A}{element}")

    def _format_dataframe_with_checkboxes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format DataFrame to replace boolean values with checkboxes.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format.

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with checkboxes.
        """
        formatted_df = df.copy()
        checkbox_columns = ["Observed", "Not Observed", "Maybe Observed"]
        for col in checkbox_columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: "✔" if x else "✘")
        return formatted_df

    def _target_abundance(self, target: str | Iterable[str]) -> dict[str, float]:
        """
        Get natural abundance of isotopes in target material.

        Parameters
        ----------
        target : str | Iterable[str]
            Target material or list of isotopes.

        Returns
        -------
        dict[str, float]
            Dictionary mapping isotope names to abundances (as fractions).

        Raises
        ------
        TypeError
            If target has invalid type.
        """
        if isinstance(target, str):
            element = ci.Element(target)
            return {
                name: abundance / 100
                for name, abundance in zip(
                    element.abundances["isotope"], element.abundances["abundance"]
                )
            }
        elif isinstance(target, Iterable):
            isotope_data = {}
            for iso in target:
                iso_obj = ci.Isotope(iso.upper())
                isotope_data[iso_obj._short_name] = iso_obj.abundance / 100
            return isotope_data
        else:
            raise TypeError(f"Target must be str or Iterable[str], got {type(target)}")

    def _load_npy(self, path: Path) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Load cross-section data from .npy file.

        Parameters
        ----------
        path : Path
            Path to .npy file.

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64]]
            Energy and cross-section arrays.

        Raises
        ------
        FileNotFoundError
            If file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        data = np.load(path)
        return data[0], data[1]