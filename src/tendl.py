import requests
import numpy as np
import matplotlib.pyplot as plt
from numpy import float64, int64
from numpy.typing import NDArray, ArrayLike
from scipy.interpolate import splev, splrep


class Tendl:
    """
    A class for retrieving, processing, and plotting nuclear reaction cross-section data from the TENDL-2023 database.

    This class handles:
    - Conversion of nuclide names to TENDL format.
    - Retrieval and interpolation of cross-section data, including decay chain contributions.
    - URL construction for TENDL file access.
    - Data formatting, scaling by target abundance, and plotting.

    Attributes
    ----------
    target : dict[str, float]
        Target composition, where keys are nuclide names (TENDL format) and values are their abundances.
    beamParticle : str
        Incident beam particle ('proton', 'deuteron', 'alpha').

    Methods
    -------
    `__init__`(target: dict[str, float], beamParticle: str) -> None:
        Initialize the Tendl object with target composition and beam particle.
    `tendl_data`(productZ: str, productA: str, isomer_level: str | None = None, Elimit: float | None = None) -> tuple[ArrayLike, ArrayLike]:
        Retrieve and interpolate TENDL cross-section data for a given isotope, with optional isomer level and energy limit, across all targets.
    `plot_tendl23_unique`(productZ: str, productA: str, Elimit: float | None = None, isomer_level: str | None = None, color: str = 'blue', lineStyle: str = '--', label: str = 'TENDL-2023') -> None:
        Retrieve and plot TENDL-2023 cross-section data for a given isotope, with options for isomer level, energy limit, and plot style.
    `plot_data_with_multiple_feeding`(productZ: str, productA: str, isomer_level: str, betaPlusDecayChain: dict[str, tuple[str, float, str]] | None = None, betaMinusDecayChain: dict[str, tuple[str, float, str]] | None = None, isomerDecayChain: dict[str, tuple[float, str]] | None = None) -> None:
        Plot cross-section data for an isotope, including summed contributions from beta and isomeric decay chains
    `_retrieve_tendl_data`(productZ: str, productA: str, isomer_level: str | None = None, Elimit: float | None = None) -> tuple[ArrayLike, ArrayLike]:
        Retrieve and interpolate TENDL cross-section data for a given isotope.
    `_get_weighted_cs`(chain, productZ, productA, is_isomer=False) -> list[ArrayLike]:
        Calculate weighted cross-sections for decay chains.
    `_tendl_url`(target_foil: str, target: str, product: str, file_ending: str, beam_type: str | None = None) -> str:
        Construct the URL for accessing TENDL nuclear data files based on the specified parameters.
    """

    # Class constants
    TENDL_HEADER_LINES = 27  # Number of header lines to skip in TENDL files
    ISOTOPE_ID_LENGTH = 3  # Required length for isotope identifiers
    TWO_DIGIT_ISOTOPE_LENGTH = 2  # Length threshold for adding leading zero

    target: dict[str, float]
    beamParticle: str

    def __init__(self, target: dict[str, float], beamParticle: str) -> None:
        """
        Initialize the Tendl object with a target composition and a beam particle.

        Parameters
        ----------
        target : dict[str, float]
            Dictionary of nuclide names and their abundances. Names are converted to TENDL format internally.
        beamParticle : str
            Name of the incident beam particle.

        Notes
        -----
        The keys of the `target` dictionary are converted to TENDL format using the `_name_trans_curie_tendl` method.
        """
        self.target = {
            self._name_trans_curie_tendl(k): v for k, v in target.items()
        }  # Convert keys to TENDL format
        self.beamParticle = beamParticle

    def tendl_data(
        self,
        productZ: str,
        productA: str,
        isomer_level: str | None = None,
        Elimit: float | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Retrieve and interpolate TENDL cross-section data for a given isotope, with optional isomer level and energy limit, across all targets.

        Parameters
        ----------
        productZ : str
            Atomic number (Z) of the product isotope as a string.
        productA : str
            Mass number (A) of the product isotope as a string.
        isomer_level : str or None, optional
            Isomeric state of the product isotope (e.g., 'm1', 'm2'). If None, ground state is assumed.
        Elimit : float or None, optional
            Upper energy limit for the data interpolation. If None, no energy limit is applied.

        Returns
        -------
        tuple[list[NDArray], list[NDArray]]
            Containing two lists:
            - List of numpy arrays with energy values for each target isotope.
            - List of numpy arrays with corresponding cross-section values for each target isotope.

        Raises
        ------
        Exception
            If no data is found for the specified target and product isotope combination.
        """
        return self._retrieve_tendl_data(productZ, productA, isomer_level, Elimit)

    def plot_tendl23_unique(
        self,
        productZ: str,
        productA: str,
        Elimit: float | None = None,
        isomer_level: str | None = None,
        color: str = "blue",
        lineStyle: str = "--",
        label: str = "TENDL-2023",
    ) -> None:
        """
        Retrieves and plots TENDL-2023 cross-section data for a given isotope, with options for isomer level, energy limit, and plot style.

        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product nucleus as a string.
        productA : str
            The mass number (A) of the product nucleus as a string.
        Elimit : float or None, optional
            The upper energy limit for the data to be plotted. If None, all available energies are used.
        isomer_level : str or None, optional
            The isomeric level of the product nucleus. If None, the ground state is assumed.
        color : str, default='blue'
            The color of the plot line.
        lineStyle : str, default='--'
            The style of the plot line (e.g., '--', '-').
        label : str, default='TENDL-2023'
            The label for the plot legend.

        Returns
        -------
        None
            This method produces a plot.

        Raises
        ------
        Exception
            If the TENDL data cannot be retrieved (e.g., due to lack of internet connection), an error message is printed.
        """
        try:
            E, Cs = self.tendl_data(productZ, productA, isomer_level, Elimit)
            plt.plot(E, Cs, label=label, linestyle=lineStyle, color=color)
        except Exception as e:
            print(f"Unable to retrieve TENDL data: {e}")

    def plot_data_with_multiple_feeding(
        self,
        productZ: str,
        productA: str,
        isomer_level: str,
        betaPlusDecayChain: dict[str, tuple[str, float, str]] | None = None,
        betaMinusDecayChain: dict[str, tuple[str, float, str]] | None = None,
        isomerDecayChain: dict[str, tuple[float, str]] | None = None,
    ) -> None:
        """
        Plots cross-section data for an isotope, including summed contributions from beta and isomeric decay chains.

        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product isotope as a string.
        productA : str
            The mass number (A) of the product isotope as a string.
        isomer_level : str
            The isomeric level of the product isotope as a string.
        betaPlusDecayChain : dict[str, tuple[str, float, str]], optional
            Dictionary mapping isotope identifiers to tuples containing (productZ, branchingRatio, isomer_level) for beta-plus decay chains.
        betaMinusDecayChain : dict[str, tuple[str, float, str]], optional
            Dictionary mapping isotope identifiers to tuples containing (productZ, branchingRatio, isomer_level) for beta-minus decay chains.
        isomerDecayChain : dict[str, tuple[float, str]], optional
            Dictionary mapping isotope identifiers to tuples containing (branchingRatio, isomer_level) for isomeric transitions.

        Returns
        -------
        None
            This method generates a plot of the total cross-section.
        """
        try:
            E, Cs = self.tendl_data(productZ, productA, isomer_level)

            # Collect all decay contributions
            all_contributions = []
            all_contributions.extend(
                self._get_weighted_cs(betaPlusDecayChain, productZ, productA)
            )
            all_contributions.extend(
                self._get_weighted_cs(betaMinusDecayChain, productZ, productA)
            )
            all_contributions.extend(
                self._get_weighted_cs(
                    isomerDecayChain, productZ, productA, is_isomer=True
                )
            )

            # Sum contributions
            Cs_tot = Cs
            if all_contributions:
                Cs_tot += np.sum(all_contributions, axis=0)

            plt.plot(E, Cs_tot, label="TENDL-2023", linestyle="--", color="blue")

        except Exception as e:
            print(f"Unable to retrieve TENDL data: {e}")

    def _retrieve_tendl_data(
        self,
        productZ: str,
        productA: str,
        isomer_level: str | None = None,
        Elimit: float | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Retrieve and interpolate TENDL cross-section data for a given isotope.

        Parameters
        ----------
        productZ : str
            Atomic number (Z) of the product isotope as a string.
        productA : str
            Mass number (A) of the product isotope as a string.
        isomer_level : str or None, optional
            Isomeric state of the product isotope (e.g., 'm1', 'm2'). If None, ground state is assumed.
        Elimit : float or None, optional
            Upper energy limit for the data interpolation. If None, no energy limit is applied.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Interpolated energy and cross-section data for the specified isotope.

        Raises
        ------
        Exception
            If no data is found for the specified target and product isotope combination.
        """
        foil = list(self.target.keys())[0][:2]
        product = self._format_product(productZ, productA)
        file_ending = self._tendl_file_ending(isomer_level)
        E, Cs = [], []

        # Retrieve cross-section data from each target isotope
        for t in self.target:
            data = self._retrieve_tendl_data_from_url(
                self._tendl_url(foil, t, product, file_ending), t
            )
            if all(isinstance(d, np.ndarray) and len(d) > 0 for d in data):
                E.append(data[0])
                Cs.append(data[1])

        if not E or not Cs:
            raise Exception(
                f"TENDL: No data for {foil} with Z={productZ}, A={productA}"
            )

        CsSum = np.sum(Cs, axis=0)
        E_interp, Cs_interp = Tools().interpolate(E[0], CsSum, xlimit=Elimit)

        return E_interp, Cs_interp

    def _get_weighted_cs(
        self,
        chain: dict[str, tuple] | dict[str, tuple] | None,
        productZ: str,
        productA: str,
        is_isomer: bool = False,
    ) -> list[ArrayLike]:
        """
        Calculate weighted cross-sections for decay chains.

        Parameters
        ----------
        chain : dict or None
            If is_isomer is True: dict[str, tuple[float, str]]
                Dictionary mapping isotope identifiers to (branchingRatio, isomer_level) for isomeric decay chains.
            If is_isomer is False: dict[str, tuple[str, float, str]]
                Dictionary mapping isotope identifiers to (productZ, branchingRatio, isomer_level) for beta decay chains.
        productZ : str
            Atomic number of the main product isotope.
        productA : str
            Mass number of the product isotope.
        is_isomer : bool, default False
            Whether this is an isomer decay chain.

        Returns
        -------
        list[ArrayLike]
            List of weighted cross-section arrays.
        """
        if not chain:
            return []

        results = []
        for _, vals in chain.items():
            if is_isomer:
                branchingRatio, isomer_level = vals
                _, Cs_val = self.tendl_data(productZ, productA, isomer_level)
            else:
                Z, branchingRatio, isomer_level = vals
                _, Cs_val = self.tendl_data(Z, productA, isomer_level)
            results.append(Cs_val * branchingRatio)

        return results

    def _name_trans_curie_tendl(self, name: str) -> str:
        """
        Converts a nuclide name from Curie to TENDL format (e.g., 'AG108' -> 'Ag108').

        Parameters
        ----------
        name : str
            The nuclide name in Curie notation (e.g., 'AG108').

        Returns
        -------
        str
            The nuclide name in TENDL notation (e.g., 'Ag108').
        """
        digits = "".join(filter(str.isdigit, name))
        symbol = "".join(filter(str.isalpha, name)).capitalize()
        return f"{symbol}{digits}"

    def _format_product(self, productZ: str, productA: str) -> str:
        """
        Formats the productZ and productA strings by ensuring each is at least three characters long,
        padding with a leading zero if necessary, and concatenates them.

        Parameters
        ----------
        productZ : str
            Atomic number as string.
        productA : str
            Mass number as string.

        Returns
        -------
        str
            Concatenated zero-padded string.
        """
        productZ = productZ.zfill(self.ISOTOPE_ID_LENGTH)
        productA = productA.zfill(self.ISOTOPE_ID_LENGTH)
        return productZ + productA

    def _tendl_url(
        self,
        target_foil: str,
        target: str,
        product: str,
        file_ending: str,
        beam_type: str | None = None,
    ) -> str:
        """
        Constructs the URL for accessing TENDL nuclear data files based on the specified parameters.

        Parameters
        ----------
        target_foil : str
            The foil identifier or name for the target material (e.g., 'Ag').
        target : str
            The target isotope or element symbol (e.g., 'Ag107').
        product : str
            The product identifier or code for the nuclear reaction.
        file_ending : str
            The file extension or ending for the desired data file.
        beam_type : str, optional
            The type of beam particle (e.g., 'deuteron', 'proton', 'alpha'). If None, defaults to the beam particle specified in the class.

        Returns
        -------
        str
            The constructed URL string pointing to the requested TENDL data file.

        Raises
        ------
        Exception
            If the beam particle type is invalid (not 'deuteron', 'proton', or 'alpha').
        """
        beam_type = beam_type or self.beamParticle

        # Map beam particle types to their TENDL directory paths
        beam_map = {
            "deuteron": "deuteron_file/",
            "proton": "proton_file/",
            "alpha": "alpha_file/",
        }

        if beam_type not in beam_map:
            raise ValueError(
                f"Invalid beam particle: {beam_type}. Must be deuteron, proton, or alpha."
            )

        target = self._format_target_length(target_foil, target)
        return f"https://tendl.web.psi.ch/tendl_2023/{beam_map[beam_type]}{target_foil}/{target}/tables/residual/rp{product}{file_ending}"

    def _format_target_length(self, target_foil: str, target_isotope: str) -> str:
        """
        Format isotope number to three digits by perprending zero if needed.
        E.g. 'Cu65' -> 'Cu065'.

        Parameters
        ----------
        target_foil : str
            The chemical element symbol (e.g., 'Cu' for copper).
        target_isotope : str
            The full isotope string, consisting of the element symbol followed by the isotope number (e.g., 'Cu65').

        Returns
        -------
        str
            The formatted target string with a three-digit isotope number (e.g., 'Cu065').
        """
        isotopeNumber = target_isotope[len(target_foil) :]
        if len(isotopeNumber) == self.TWO_DIGIT_ISOTOPE_LENGTH:
            isotopeNumber = "0" + isotopeNumber
        return target_foil + isotopeNumber

    def _tendl_file_ending(self, isomer_level: str | None = None) -> str:
        """
        Return the TENDL file suffic based on isomer level.

        If `isomer_level` is None, returns the default '.tot' suffix.
        Otherwise, returns '.L{isomer_level}' to specify the isomer state.

        Parameters
        ----------
        isomer_level : str or None, optional
            The isomer level to include in the file ending (e.g., '01', '02'). If None, the default ending is used.

        Returns
        -------
        str
            The appropriate file ending for the TENDL file.
        """
        return f".tot" if isomer_level is None else f".L{isomer_level}"

    def _retrieve_tendl_data_from_url(
        self, url: str, target: str
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Downloads TENDL nuclear data for a target and returns energy and scaled cross-section arrays.

        Parameters
        ----------
        url : str
            The URL pointing to the TENDL data file.
        target : str
            The key identifying the target isotope in the `self.target` dictionary.

        Returns
        -------
        tuple[NDArray, NDArray]
            Two arrays::
            - E: Energy values from the TENDL data.
            - Cs: Cross-section values scaled by the target abundance.

        Notes
        -----
        If the data cannot be retrieved or parsed, empty numpy arrays are returned.
        """
        try:
            tendlData = requests.get(url).text.split("\n")[
                self.TENDL_HEADER_LINES :
            ]  # Skip header lines
            tendlData = np.genfromtxt(tendlData)
            abundance = self.target[target]
            E = tendlData[:, 0]
            Cs = tendlData[:, 1]

            # Scale cross-sections by target isotope abundance
            return E, Cs * abundance

        except Exception as e:
            print(f"Unable to retrieve tendlData from url: {url} - {e}")
            return np.array([]), np.array([])


class Tools:
    """
    Provides utility methods for B-spline interpolation with optional zero padding and x-axis limiting.


    Methods
    -------
    `interpolate`(x: ArrayLike, y: ArrayLike, xlimit: float | None = None, zero_padding: bool = False) -> tuple[ArrayLike, ArrayLike]:
           Performs B-spline interpolation (degree 5) on input data, with optional padding and x-range limit.

    `zero_padding`(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[float64 | int64], NDArray[float64 | int64]]:
        Prepends zeros to `x` and `y` if `x` does not start at 0.
    """

    # Class constants
    DEFAULT_INTERPOLATION_POINTS = 1000  # Default number of points for interpolation
    ZERO_PADDING_POINTS = 10  # Number of points to add for zero padding
    ZERO_PADDING_OFFSET = 0.5  # Offset for zero padding range

    def interpolate(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xlimit: float | None = None,
        zero_padding: bool = False,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Interpolates input data using a degree-5 B-spline.

        Optionally pads the input with zeros and restricts interpolation to a specified x-range.

        Parameters
        ----------
        x : ArrayLike
            Input x-values.
        y : ArrayLike
            Input y-values.
        xlimit : float, optional
            Upper limit for the x-axis; if set, output x-values span [1, xlimit] with 1000 points.
        zero_padding : bool, default False
            If True, prepends zeros to input arrays before interpolation.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Interpolated x and y values.
        """
        if zero_padding:
            x, y = self.zero_padding(x, y)

        tck = splrep(x, y, s=0, k=5)  # B-spline fit, degree 5, no smoothing

        if xlimit:
            x = np.linspace(1, xlimit, self.DEFAULT_INTERPOLATION_POINTS)

        y = splev(x, tck, der=0)  # Evaluate B-spline at x
        return x, y

    def zero_padding(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[float64 | int64], NDArray[float64 | int64]]:
        """
        Prepends zeros to `x` and `y` if `x` does not start at 0.

        Adds 10 evenly spaced x-values from 0 to `x[0] - 0.5`, with corresponding y-values of 0.

        Parameters
        ----------
        x : ArrayLike
            Input x-values.
        y : ArrayLike
            Input y-values.

        Returns
        -------
        tuple of numpy.ndarray
            Zero-padded x and y arrays.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if x[0] != 0:
            x_pad = np.linspace(
                0, x[0] - self.ZERO_PADDING_OFFSET, self.ZERO_PADDING_POINTS
            )
            y_pad = np.zeros_like(x_pad)
            x = np.concatenate((x_pad, x))
            y = np.concatenate((y_pad, y))

        return x, y