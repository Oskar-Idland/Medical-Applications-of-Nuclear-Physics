from urllib.request import urlopen

import numpy as np
import requests
import matplotlib.pyplot as plt
from numpy.typing import NDArray, ArrayLike
from numpy import float64, int64
from scipy.interpolate import splev, splrep


class Tendl:
    """
    Tendl class for retrieving and processing nuclear reaction cross-section data from the TENDL database.
    This class provides methods to fetch, interpolate, and plot nuclear data (such as cross-sections) for specified target compositions and beam particles using the TENDL-2023 database. It supports handling different reaction products, isomeric states, and decay chains, and allows for plotting the resulting data using matplotlib.
        The name of the incident beam particle (e.g., 'deuteron', 'proton', 'alpha').
    Attributes
        Target composition with keys converted to TENDL format.
    Methods
    _name_trans_curie_tendl(name: str) -> str
    _tendlDeuteronData(productZ: str, productA: str, isomerLevel: str | None = None) -> tuple[list[NDArray], list[NDArray]]
    tendlData(productZ: str, productA: str, isomerLevel: str | None = None, Elimit: float | None = None) -> tuple[list[NDArray], list[NDArray]]
    plotTendl23(productZ: str, productA: str, isomerLevel: str | None = None) -> None
    plotTendl23Unique(productZ: str, productA: str, Elimit: float | None = None, isomerLevel: str | None = None, color: str = 'blue', lineStyle: str = '--', label: str = 'TENDL-2023') -> None
        Plots the TENDL-2023 cross-section data for a specified nuclear reaction product with customizable plot appearance.
    plotdataWithMultipleFeeding(productZ: str, productA: str, isomerLevel: str, betaPlusDecayChain: dict[str, tuple[str, float, str]] | None = None, betaMinusDecayChain: dict[str, tuple[str, float, str]] | None = None, isomerDecayChain: dict[str, tuple[float, str]] | None = None) -> None
    _product(productZ: str, productA: str) -> str
        Formats the productZ and productA strings and concatenates them for TENDL queries.
    _tendDeuteronlUrl(targetFoil: str, target: str, product: str, fileEnding: str) -> str
    _tendlUrl(targetFoil: str, target: str, product: str, fileEnding: str) -> str
    _formatTargetLength(targetFoil: str, targetIsotope: str) -> str
    _tendlFileEnding(isomerLevel: str | None = None) -> str
    _retrieveTendlDataFromUrl(url: str, target: str) -> tuple[NDArray, NDArray]
    _retrieveDataFromUrlWithNumpy(url: str) -> tuple[NDArray, NDArray]
    """
    def __init__(self, target: dict[str, float], beamParticle: str) -> None:
        """
        Initialize the object with target composition and beam particle.
        
        Parameters
        ----------
        target : dict[str, float]
            Dictionary representing the target composition, where keys are element or isotope names and values are their respective abundances or fractions.
        beamParticle : str
            The name of the incident beam particle.
        
        Notes
        -----
        The keys of the `target` dictionary are converted to TENDL format using the `_name_trans_curie_tendl` method.
        """
        
        self.target = {self._name_trans_curie_tendl(k): v for k, v in target.items()}  # Convert keys to TENDL format
        self.beamParticle = beamParticle
    
    def _name_trans_curie_tendl(self, name: str) -> str:
        """
        Transforms a nuclide name from Curie notation to TENDL notation.
        This function takes a nuclide name string (e.g., 'AG108') and converts it to the TENDL format,
        where the element symbol is capitalized properly and the mass number follows (e.g., 'Ag108').
        
        Parameters
        ----------
        name : str
            The nuclide name in Curie notation (e.g., 'AG108').
        
        Returns
        -------
        str
            The nuclide name in TENDL notation (e.g., 'Ag108').
        """

        numbers_part = ''.join(filter(str.isdigit, name))
        letters_part = ''.join(filter(str.isalpha, name)).capitalize()  # Extract the letters and capitalize (e.g., 'AG' -> 'Ag')
        return f"{letters_part}{numbers_part}"
    






    def _tendlDeuteronData(self, productZ: str, productA: str, isomerLevel: str | None = None) -> tuple[list[NDArray], list[NDArray]]:
        """
        Retrieves and processes TENDL deuteron data for a specified product isotope and optional isomer level.
        This method fetches cross-section data for deuteron-induced reactions from TENDL, interpolates the data,
        and returns the energy and cross-section arrays for each target in the current object.
        
        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product isotope as a string.
        productA : str
            The mass number (A) of the product isotope as a string.
        isomerLevel : str or None, optional
            The isomeric state of the product isotope. If None, ground state is assumed.
        
        Returns
        -------
        tuple of list of numpy.ndarray
            A tuple containing two lists:
            - List of energy arrays (one per target).
            - List of cross-section arrays (one per target).
        
        Raises
        ------
        Exception
            If no data is found for the specified target and product isotope.
        """
        targetFoil = list(self.target.keys())[0][0:2]
        product = self._product(productZ, productA)
        fileEnding = self._tendlFileEnding(isomerLevel)
        E = []
        Cs = []
        for t in self.target.keys():
            data = self._retrieveTendlDataFromUrl(
                self._tendDeuteronlUrl(targetFoil, t, product, fileEnding), t
            )
            if isinstance(data[0], np.ndarray) and len(data[0]) > 0 and len(data[1]) > 0:
                E.append(data[0])
                Cs.append(data[1])

        if len(E) == 0 or len(Cs) == 0:
            raise Exception("TENDL: No data found for target: " + targetFoil + " for productZ" + productZ + "and product A: " + productA)

        # CsSummed = sum(np.concatenate(Cs))
        # E = E[0]
        # E, Cs = Tools().interpolate(E, CsSummed)
        E, Cs = zip(*(Tools().interpolate(x=E_i, y=Cs_i) for E_i, Cs_i in zip(E, Cs)))

        return E, Cs
    



    def tendlData(self, productZ: str, productA: str, isomerLevel: str | None = None, Elimit: float | None = None) -> tuple[list[NDArray], list[NDArray]]:
        """
        Retrieve and interpolate TENDL nuclear data for a specified product isotope.

        This method fetches cross-section data for a given product isotope (specified by atomic number and mass number)
        from the TENDL database, optionally for a specific isomeric state and up to a specified energy limit. The data
        is retrieved for all available target isotopes, interpolated as needed, and returned as lists of energy and
        cross-section arrays.

        Parameters
        ----------
        productZ : str
            Atomic number (Z) of the product isotope as a string.
        productA : str
            Mass number (A) of the product isotope as a string.
        isomerLevel : str or None, optional
            Isomeric state of the product isotope (e.g., 'm1', 'm2'). If None, ground state is assumed.
        Elimit : float or None, optional
            Upper energy limit for the data interpolation. If None, no energy limit is applied.

        Returns
        -------
        tuple[list[numpy.ndarray], list[numpy.ndarray]]
            A tuple containing two lists:
            - List of numpy arrays with energy values for each target isotope.
            - List of numpy arrays with corresponding cross-section values for each target isotope.

        Raises
        ------
        Exception
            If no data is found for the specified target and product isotope combination.
        """
        targetFoil = list(self.target.keys())[0][0:2]
        product = self._product(productZ, productA)
        fileEnding = self._tendlFileEnding(isomerLevel)
        E = []
        Cs = []
        for t in self.target.keys():
            data = self._retrieveTendlDataFromUrl(
                self._tendlUrl(targetFoil, t, product, fileEnding), t
            )
            if isinstance(data[0], np.ndarray) and len(data[0]) > 0 and len(data[1]) > 0:
                E.append(data[0])
                Cs.append(data[1])

        if len(E) == 0 or len(Cs) == 0:
            raise Exception("TENDL: No data found for target: " + targetFoil + " for productZ" + productZ + "and product A: " + productA)
        
        # CsSummed = sum(Cs)
        # E = E[0]
        # E, Cs = Tools().interpolate(x=E, y=CsSummed, xlimit=Elimit)
        
        # x, y, xlimit=None, zeroPadding=False
        # print(Elimit)

        E, Cs = zip(*(Tools().interpolate(x=E_i, y=Cs_i, xlimit=Elimit) for E_i, Cs_i in zip(E, Cs)))

        return list(E), list(Cs)
    
    def plotTendl23(self, productZ: str, productA: str, isomerLevel: str | None = None) -> None:  # , feeding = None, branchingRatio = None, parentIsomerLevel = None):
        """
        Plots the TENDL-2023 cross-section data for a specified nuclear reaction product.

        This method retrieves and plots the cross-section data for a given product defined by its atomic number (Z), mass number (A), and optional isomeric level. The data is visualized using a dashed blue line labeled 'TENDL-2023'.

        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product as a string.
        productA : str
            The mass number (A) of the product as a string.
        isomerLevel : str or None, optional
            The isomeric level of the product, if applicable. Default is None.

        Returns
        -------
        None
            This method does not return any value. It generates a plot as a side effect.
        """
        # try:
        E, Cs = self._tendlDeuteronData(productZ, productA, isomerLevel)
        # if feeding == 'beta+' or feeding == 'beta-':
        # CsParent = self.correctForFeeding(productZ, productA, feeding, branchingRatio, parentIsomerLevel)[1]
        # Cs = Cs + CsParent
        plt.plot(E, Cs, label='TENDL-2023', linestyle='--', color='blue')

    # except:
    # print("Unable to retrive tendl data, perhaps no internet connection?")


    def plotTendl23Unique(self, productZ: str, productA: str, Elimit: float | None = None, isomerLevel: str | None = None, color: str = 'blue', lineStyle: str = '--', label: str = 'TENDL-2023') -> None:
        """
        Plots the TENDL-2023 cross-section data for a specified nuclear reaction product.
        This method retrieves cross-section data for a given product (specified by atomic number and mass number)
        from the TENDL-2023 database and plots it using matplotlib. Optional parameters allow for filtering by
        isomeric level and energy limit, as well as customizing the plot's appearance.
        
        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product nucleus as a string.
        productA : str
            The mass number (A) of the product nucleus as a string.
        Elimit : float or None, optional
            The upper energy limit for the data to be plotted. If None, all available energies are used.
        isomerLevel : str or None, optional
            The isomeric level of the product nucleus. If None, the ground state is assumed.
        color : str, default='blue'
            The color of the plot line.
        lineStyle : str, default='--'
            The style of the plot line (e.g., '--', '-', '-.', ':').
        label : str, default='TENDL-2023'
            The label for the plot legend.
        
        Returns
        -------
        None
            This method does not return any value. It produces a plot as a side effect.
        
        Raises
        ------
        Exception
            If the TENDL data cannot be retrieved (e.g., due to lack of internet connection), an error message is printed.
        """
        try:
            E, Cs = self.tendlData(productZ, productA, isomerLevel, Elimit)
            plt.plot(E, Cs, label=label, linestyle=lineStyle, color=color)
        except:
            print("Unable to retrive tendl data, perhaps no internet connection?")










    # def plotdataWithMultipleFeeding(self, productZ: str, productA: str, isomerLevel: str, betaPlusDecayChain: dict[str, tuple[str, float, str]] | None = None, betaMinusDecayChain: dict[str, tuple[str, float, str]] | None = None, isomerDecayChain: dict[str, tuple[float, str]] | None = None) -> None:
    #     # {isotope: (productZ, branchingRatio, isomerLevel)} #beta+/beta-
    #     # {isotope: (branchingRatio, isomerLevel)} #isomer
    #     try:
    #         E, Cs = self._tendlDeuteronData(productZ, productA, isomerLevel)
    #         Cs_betaplus = []
    #         Cs_betaminus = []
    #         Cs_isomer = []

    #         if betaPlusDecayChain:
    #             for i in list(betaPlusDecayChain.keys()):
    #                 Z = betaPlusDecayChain[i][0]
    #                 branchingRatio = betaPlusDecayChain[i][1]
    #                 isomerLevel = betaPlusDecayChain[i][2]
    #                 E_bp, Cs_bp = self._tendlDeuteronData(Z, productA, isomerLevel)
    #                 Cs_betaplus.append(Cs_bp * branchingRatio)

    #         if betaMinusDecayChain:
    #             for i in list(betaMinusDecayChain.keys()):
    #                 Z = betaMinusDecayChain[i][0]
    #                 branchingRatio = betaMinusDecayChain[i][1]
    #                 isomerLevel = betaMinusDecayChain[i][2]
    #                 E_bm, Cs_bm = self._tendlDeuteronData(Z, productA, isomerLevel)
    #                 Cs_betaminus.append(Cs_bm * branchingRatio)

    #         if isomerDecayChain:
    #             for i in list(isomerDecayChain.keys()):
    #                 branchingRatio = isomerDecayChain[i][0]
    #                 isomerLevel = isomerDecayChain[i][1]
    #                 E_i, Cs_i = self._tendlDeuteronData(productZ, productA, isomerLevel)
    #                 Cs_isomer.append(Cs_i * branchingRatio)

    #         totCs = Cs + sum(Cs_betaplus) + sum(Cs_betaminus) + sum(Cs_isomer)
    #         plt.plot(E, totCs, label='TENDL-2023', linestyle='--', color='blue')

    #     except:
    #         print("Unable to retrive tendl data, perhaps no internet connection?")


    def plotdataWithMultipleFeeding(self, productZ: str, productA: str, isomerLevel: str, betaPlusDecayChain: dict[str, tuple[str, float, str]] | None = None, betaMinusDecayChain: dict[str, tuple[str, float, str]] | None = None, isomerDecayChain: dict[str, tuple[float, str]] | None = None) -> None:
        """
        Plots cross-section data for a specified isotope, including contributions from multiple decay chains (beta-plus, beta-minus, and isomeric transitions).
        This method retrieves and sums cross-section data for the main isotope and any specified feeding from decay chains, then plots the total cross-section as a function of energy.
        
        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product isotope as a string.
        productA : str
            The mass number (A) of the product isotope as a string.
        isomerLevel : str
            The isomeric level of the product isotope as a string.
        betaPlusDecayChain : dict[str, tuple[str, float, str]], optional
            Dictionary mapping isotope identifiers to tuples containing (productZ, branchingRatio, isomerLevel) for beta-plus decay chains.
        betaMinusDecayChain : dict[str, tuple[str, float, str]], optional
            Dictionary mapping isotope identifiers to tuples containing (productZ, branchingRatio, isomerLevel) for beta-minus decay chains.
        isomerDecayChain : dict[str, tuple[float, str]], optional
            Dictionary mapping isotope identifiers to tuples containing (branchingRatio, isomerLevel) for isomeric transitions.
        
        Returns
        -------
        None
            This method does not return a value. It generates a plot of the total cross-section.
        """
        # {isotope: (productZ, branchingRatio, isomerLevel)} #beta+/beta-
        # {isotope: (branchingRatio, isomerLevel)} #isomer
        try:
            E, Cs = self._tendlDeuteronData(productZ, productA, isomerLevel)
            Cs_total = np.sum(Cs, axis=0)

            Cs_betaplus = []
            Cs_betaminus = []
            Cs_isomer = []

            if betaPlusDecayChain:
                for i in betaPlusDecayChain:
                    Z, branchingRatio, isomerLevel = betaPlusDecayChain[i]
                    _, Cs_bp_list = self._tendlDeuteronData(Z, productA, isomerLevel)
                    Cs_bp = np.sum(Cs_bp_list, axis=0)
                    Cs_betaplus.append(Cs_bp * branchingRatio)

            if betaMinusDecayChain:
                for i in betaMinusDecayChain:
                    Z, branchingRatio, isomerLevel = betaMinusDecayChain[i]
                    _, Cs_bm_list = self._tendlDeuteronData(Z, productA, isomerLevel)
                    Cs_bm = np.sum(Cs_bm_list, axis=0)
                    Cs_betaminus.append(Cs_bm * branchingRatio)

            if isomerDecayChain:
                for i in isomerDecayChain:
                    branchingRatio, isomerLevel = isomerDecayChain[i]
                    _, Cs_i_list = self._tendlDeuteronData(productZ, productA, isomerLevel)
                    Cs_i = np.sum(Cs_i_list, axis=0)
                    Cs_isomer.append(Cs_i * branchingRatio)

            if Cs_betaplus:
                Cs_total += np.sum(Cs_betaplus, axis=0)
            if Cs_betaminus:
                Cs_total += np.sum(Cs_betaminus, axis=0)
            if Cs_isomer:
                Cs_total += np.sum(Cs_isomer, axis=0)
            
            plt.plot(E[0], Cs_total, label='TENDL-2023', linestyle='--', color='blue')

        except Exception as e:
            print("Unable to retrive tendl data, perhaps no internet connection?", e)


    


       

    

    

    

    def _product(self, productZ: str, productA: str) -> str:
        """
        Formats the productZ and productA strings by ensuring each is at least three characters long,
        padding with a leading zero if necessary, and concatenates them.
        
        Parameters
        ----------
        productZ : str
            The atomic number (Z) of the product as a string.
        productA : str
            The mass number (A) of the product as a string.
        
        Returns
        -------
        str
            The concatenated and zero-padded string of productZ and productA.
        """
        if len(productZ) <= 2:
            productZ = '0' + productZ
        else:
            productZ = productZ

        if len(productA) <= 2:
            productA = '0' + productA
        else:
            productA = productA

        return productZ + productA

    def _tendDeuteronlUrl(self, targetFoil: str, target: str, product: str, fileEnding: str) -> str:
        """
        Constructs a URL for accessing deuteron-induced nuclear reaction data from the TENDL database.
        If the `target` string is less than 5 characters, a '0' is inserted after the first two characters to ensure proper formatting.
        
        Parameters
        ----------
        targetFoil : str
            The name of the target foil (e.g., 'Ag', 'Cu').
        target : str
            The target isotope identifier, typically in the format 'Ag107' or similar.
        product : str
            The product identifier for the nuclear reaction.
        fileEnding : str
            The file extension or ending for the URL (e.g., '.txt').
        
        Returns
        -------
        str
            The constructed URL string pointing to the desired TENDL deuteron file.
        """
        if len(target) < 5:
            target = target[0:2] + '0' + target[2:]

        return (
            'https://tendl.web.psi.ch/tendl_2023/deuteron_file/'
            + targetFoil + '/' + target
            + '/tables/residual/rp'
            + product + fileEnding
        )

    def _tendlUrl(self, targetFoil: str, target: str, product: str, fileEnding: str) -> str:
        """
        Constructs the URL for accessing TENDL nuclear data files based on the specified parameters.
        
        Parameters
        ----------
        targetFoil : str
            The foil identifier or name for the target material.
        target : str
            The target isotope or element symbol.
        product : str
            The product identifier or code for the nuclear reaction.
        fileEnding : str
            The file extension or ending for the desired data file.
        
        Returns
        -------
        str
            The constructed URL string pointing to the requested TENDL data file.
        
        Raises
        ------
        Exception
            If the beam particle type is invalid (not 'deuteron', 'proton', or 'alpha').
        """
        if self.beamParticle == 'deuteron':
            beam_file = 'deuteron_file/'
        elif self.beamParticle == 'proton':
            beam_file = 'proton_file/'
        elif self.beamParticle == 'alpha':
            beam_file = 'alpha_file/'
        else:
            raise Exception("Invalid beam particle. Must be deuteron or proton. Was: " + self.beamParticle)

        target = self._formatTargetLength(targetFoil, target)

        return (
            'https://tendl.web.psi.ch/tendl_2023/'
            + beam_file
            + targetFoil + '/' + target
            + '/tables/residual/rp'
            + product + fileEnding
        )

    def _formatTargetLength(self, targetFoil: str, targetIsotope: str) -> str:
        """
        Formats the target isotope string by ensuring the isotope number is three digits.
        Given a chemical element symbol (`targetFoil`) and its isotope number (`targetIsotope`), this method appends a leading zero to the isotope number if it is only two digits, resulting in a three-digit isotope number. The formatted string is then returned as the concatenation of the element symbol and the formatted isotope number.
        
        Parameters
        ----------
        targetFoil : str
            The chemical element symbol (e.g., 'Cu' for copper).
        targetIsotope : str
            The full isotope string, consisting of the element symbol followed by the isotope number (e.g., 'Cu65').
        
        Returns
        -------
        str
            The formatted target string with a three-digit isotope number (e.g., 'Cu065').
        """
        # Cu65 --> Cu065. Ir193=Ir193
        isotopeNumber = targetIsotope[len(targetFoil):]
        formattedIsotopeNumber = isotopeNumber if len(isotopeNumber) == 3 else '0' + isotopeNumber

        return targetFoil + formattedIsotopeNumber

    def _tendlFileEnding(self, isomerLevel: str | None = None) -> str:
        """
        Generates the file ending for a TENDL file based on the isomer level.
        If no isomer level is provided, returns the default '.tot' ending.
        Otherwise, returns the ending in the format '.L{isomerLevel}'.
        
        Parameters
        ----------
        isomerLevel : str or None, optional
            The isomer level to include in the file ending. If None, the default ending is used.
        
        Returns
        -------
        str
            The appropriate file ending for the TENDL file.
        """
        return '.tot' if isomerLevel is None else '.L' + isomerLevel

    def _retrieveTendlDataFromUrl(self, url: str, target: str) -> tuple[NDArray, NDArray]:
        """
        Retrieve and process TENDL data from a given URL for a specified target.
        This method downloads TENDL nuclear data from the provided URL, skips the header lines,
        parses the data, and scales the cross-section values by the target's abundance.
        
        Parameters
        ----------
        url : str
            The URL pointing to the TENDL data file.
        target : str
            The key identifying the target isotope in the `self.target` dictionary.
        
        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing:
            - E (numpy.ndarray): The energy values extracted from the TENDL data.
            - Cs (numpy.ndarray): The cross-section values scaled by the target abundance.
        
        Notes
        -----
        If the data cannot be retrieved or parsed, empty numpy arrays are returned.
        """
        try:
            tendlData = requests.get(url).text.split("\n")[27:]  # skipping 27 first lines in tendl file
            tendlData = np.genfromtxt(tendlData)
            abundance = self.target[target]
            E = tendlData[:, 0]
            Cs = tendlData[:, 1]

            return E, Cs * abundance

        except:
            print('Unable to retrieve tendlData from url: ' + url)
            return np.array([]), np.array([])

    def _retrieveDataFromUrlWithNumpy(self, url: str) -> tuple[NDArray, NDArray]:
        """
        Fetches and parses numerical data from a given URL using NumPy.
        This method retrieves data from the specified URL, expecting a whitespace-delimited text file,
        and extracts the first two columns as energy and cross-section values.
        
        Parameters
        ----------
        url : str
            The URL pointing to the data file to be retrieved and parsed.
        
        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing two arrays:
            - energy (numpy.ndarray): The first column of the data, representing energy values.
            - xs (numpy.ndarray): The second column of the data, representing cross-section values.
        
        Raises
        ------
        URLError
            If the URL cannot be opened or accessed.
        ValueError
            If the data cannot be parsed as expected.
        """
        # tendl_data = np.genfromtxt(urlopen('ttps://tendl.web.psi.ch/tendl_2023/deuteron_file/Ir/Ir193/tables/residual/rp078193.L05', delimiter=" "))
        tendl_data = np.genfromtxt(urlopen(url), delimiter=" ")
        energy = tendl_data[:, 0]
        xs = tendl_data[:, 1]

        return energy, xs


class Tools:
    """
    A collection of utility methods for data interpolation and zero padding.
    Methods
        interpolate(x, y, xlimit=None, zeroPadding=False)
        zeroPadding(x, y)
    """
    def interpolate(self, x: ArrayLike, y: ArrayLike, xlimit: float | None = None, zeroPadding: bool = False) -> tuple[ArrayLike, ArrayLike]:
        """
        Interpolate the given data using a B-spline of degree 5, with optional zero padding and x-axis limit.
        
        Parameters
        ----------
        x : ArrayLike
            The x-coordinates of the data points to interpolate.
        y : ArrayLike
            The y-coordinates of the data points to interpolate.
        xlimit : float, optional
            The upper limit for the x-axis in the interpolation. If provided, the interpolation will be evaluated on a linspace from 1 to `xlimit` with 1000 points.
        zeroPadding : bool, default False
            If True, applies zero padding to the input data before interpolation.
        
        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            A tuple containing the interpolated x and y values.
        """
        if zeroPadding:
            x, y = self.zeroPadding(x, y)
        tck = splrep(x, y, s=0, k=5)
        if xlimit:
            x = np.linspace(1, xlimit, 1000)
        # else:
        #     x_new = np.linspace(1, 100, 1000)

        y = splev(x, tck, der=0)
        return x, y

    def zeroPadding(self, x: ArrayLike, y: ArrayLike) -> tuple[NDArray[float64 | int64], NDArray[float64 | int64]]:
        """
        Pads the input arrays `x` and `y` with zeros at the beginning if `x` does not start at 0.
        If the first element of `x` is not zero, this function prepends 10 evenly spaced values
        from 0 to just before the first value of `x` (specifically, up to `x[0] - 0.5`), and
        prepends corresponding zeros to `y`. The resulting arrays are returned.
        
        Parameters
        ----------
        x : ArrayLike
            The x-values array to be padded if necessary.
        y : ArrayLike
            The y-values array to be padded with zeros if necessary.
        
        Returns
        -------
        tuple of numpy.ndarray
            The padded x and y arrays as a tuple.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x[0] != 0:
            zero_padding = np.linspace(0, x[0] - 0.5, 10)
            zeros_y = np.zeros((len(zero_padding)))
            x = np.concatenate((zero_padding, x))
            y = np.concatenate((zeros_y, y))

        return x, y


