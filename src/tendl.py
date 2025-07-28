from urllib.request import urlopen

import numpy as np
import requests
import matplotlib.pyplot as plt
from numpy.typing import NDArray, ArrayLike
from numpy import float64, int64
from scipy.interpolate import splev, splrep


class Tendl:
    """
    ....
    
    Attributes
    ----------
        ....
    
    Methods
    -------
    ....
    """
    
    target: dict[str, float]
    beamParticle: str
    
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
    






    def _tendlDeuteronData(self, productZ: str, productA: str, isomerLevel: str | None = None) -> tuple[NDArray[float64], NDArray[float64]]:
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

        CsSummed = np.sum(Cs, axis=0)
        E = E[0]
        E, Cs = Tools().interpolate(E, CsSummed)
        print(E)
        print(Cs)

        return E, Cs
    



    def tendlData(self, productZ: str, productA: str, isomerLevel: str | None = None, Elimit: float | None = None) -> tuple[NDArray[float64], NDArray[float64]]:
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
        
        CsSummed = np.sum(Cs, axis=0)
        E = E[0]
        E, Cs = Tools().interpolate(x=E, y=CsSummed, xlimit=Elimit)
        print(E)
        print(Cs)
        
        # x, y, xlimit=None, zeroPadding=False
        # print(Elimit)

        return E, Cs
    
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
            
            Cs_betaplus = []
            Cs_betaminus = []
            Cs_isomer = []

            if betaPlusDecayChain:
                for i in betaPlusDecayChain:
                    Z, branchingRatio, isomerLevel = betaPlusDecayChain[i]
                    _, Cs_bp = self._tendlDeuteronData(Z, productA, isomerLevel)
                    Cs_betaplus.append(Cs_bp * branchingRatio)

            if betaMinusDecayChain:
                for i in betaMinusDecayChain:
                    Z, branchingRatio, isomerLevel = betaMinusDecayChain[i]
                    _, Cs_bm = self._tendlDeuteronData(Z, productA, isomerLevel)
                    Cs_betaminus.append(Cs_bm * branchingRatio)

            if isomerDecayChain:
                for i in isomerDecayChain:
                    branchingRatio, isomerLevel = isomerDecayChain[i]
                    _, Cs_i = self._tendlDeuteronData(productZ, productA, isomerLevel)
                    Cs_isomer.append(Cs_i * branchingRatio)

            Cs_tot = Cs + sum(Cs_betaplus) + sum(Cs_betaminus) + sum(Cs_isomer)
            
            plt.plot(E, Cs_tot, label='TENDL-2023', linestyle='--', color='blue')

        except Exception as e:
            print("Unable to retrive tendl data, perhaps no internet connection?", e)

    def _product(
        self,
        productZ: str,
        productA: str
    ) -> str:
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
        productZ = productZ.zfill(3)  
        productA = productA.zfill(3)
        return productZ + productA

    def _tendDeuteronlUrl(
        self,
        targetFoil: str,
        target: str,
        product: str,
        fileEnding: str
    ) -> str:
        """
        Constructs a URL for accessing deuteron-induced nuclear reaction data from the TENDL database.
        Pads `target` isotope with '0' if shorter than 5 characters.
        
        Parameters
        ----------
        targetFoil : str
            The foil name name (e.g., 'Ag').
        target : str
            Target isotope ID (e.g., 'Ag107').
        product : str
            The product identifier for the nuclear reaction.
        fileEnding : str
            File extension or ending for the URL (e.g., '.txt').
        
        Returns
        -------
        str
            The constructed URL string pointing to the desired TENDL deuteron file.
        """
        if len(target) < 5:
            target = target[:2] + '0' + target[2:]
        return f"https://tendl.web.psi.ch/tendl_2023/deuteron_file/{targetFoil}/{target}/tables/residual/rp{product}{fileEnding}"   

    def _tendlUrl(
        self,
        targetFoil: str,
        target: str,
        product: str,
        fileEnding: str
    ) -> str:
        """
        Constructs the URL for accessing TENDL nuclear data files based on the specified parameters.
        
        Parameters
        ----------
        targetFoil : str
            The foil identifier or name for the target material (e.g., 'Ag').
        target : str
            The target isotope or element symbol (e.g., 'Ag107').
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
        beam_map = {
            'deuteron': 'deuteron_file/',
            'proton': 'proton_file/',
            'alpha': 'alpha_file/'
        }
        try:
            beam_file = beam_map[self.beamParticle]
        except KeyError:
            raise Exception("Invalid beam particle: " + self.beamParticle + ". Must be deuteron, proton, or alpha.")
        
        target = self._formatTargetLength(targetFoil, target)

        return (
            f"https://tendl.web.psi.ch/tendl_2023/{beam_file}"
            f"{targetFoil}/{target}/tables/residual/rp{product}{fileEnding}"
        )

    def _formatTargetLength(
        self,
        targetFoil: str,
        targetIsotope: str
    ) -> str:
        """
        Format isotope number to three digits by perprending zero if needed.
        E.g. 'Cu65' → 'Cu065'.
        
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
        isotopeNumber = targetIsotope[len(targetFoil):]
        if len(isotopeNumber) == 2:
            isotopeNumber = '0' + isotopeNumber
        return targetFoil + isotopeNumber

    def _tendlFileEnding(
        self,
        isomerLevel: str | None = None
    ) -> str:
        """
        Return the TENDL file suffic based on isomer level.
        
        If `isomerLevel` is None, returns the default '.tot' suffix.
        Otherwise, returns '.L{isomerLevel}' to specify the isomer state.
        
        Parameters
        ----------
        isomerLevel : str or None, optional
            The isomer level to include in the file ending. If None, the default ending is used.
        
        Returns
        -------
        str
            The appropriate file ending for the TENDL file.
        """
        return f".tot" if isomerLevel is None else f".L{isomerLevel}"

    def _retrieveTendlDataFromUrl(
        self,
        url: str,
        target: str
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
            tendlData = requests.get(url).text.split("\n")[27:]  # skip header lines
            tendlData = np.genfromtxt(tendlData)
            abundance = self.target[target]
            E = tendlData[:, 0]
            Cs = tendlData[:, 1]

            return E, Cs * abundance

        except Exception as e:
            print(f"Unable to retrieve tendlData from url: {url} - {e}")
            return np.array([]), np.array([])

    def _retrieveDataFromUrlWithNumpy(
        self,
        url: str
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Fetches numerical data from a URL and returns first two columns as arrays: energy, cross-section
        
        Parameters
        ----------
        url : str
            The URL pointing to the data file to be retrieved and parsed.
        
        Returns
        -------
        tuple[NDArray, NDArray]
            Two arrays:
            - energy: Energy values from the first column.
            - xs: Cross-section values from the second column.

        Raises
        ------
        URLError
            If the URL cannot be opened or accessed.
        ValueError
            If the data cannot be parsed as expected.
        """
        data = np.genfromtxt(urlopen(url), delimiter=" ")

        return data[:, 0], data[:, 1]


class Tools:
    """
    Provides utility methods for B-spline interpolation with optional zero padding and x-axis limiting.


    Methods
    -------
    interpolate(x, y, xlimit=None, zeroPadding=False)
           Performs B-spline interpolation (degree 5) on input data, with optional padding and x-range limit.
    
    zeroPadding(x, y)
        Prepends zeros to `x` and `y` if `x` does not start at 0.
    """
    def interpolate(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xlimit: float | None = None,
        zeroPadding: bool = False
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
        zeroPadding : bool, default False
            If True, prepends zeros to input arrays before interpolation.
        
        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Interpolated x and y values.
        """
        if zeroPadding:
            x, y = self.zeroPadding(x, y)

        tck = splrep(x, y, s=0, k=5)  # B-spline fit, degree 5, no smoothing

        if xlimit:
            x = np.linspace(1, xlimit, 1000)

        y = splev(x, tck, der=0)  # Evaluate B-spline at x
        return x, y

    def zeroPadding(
        self,
        x: ArrayLike,
        y: ArrayLike
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
            x_pad = np.linspace(0, x[0] - 0.5, 10)
            y_pad = np.zeros_like(x_pad)
            x = np.concatenate((x_pad, x))
            y = np.concatenate((y_pad, y))

        return x, y


# if __name__ == "__main__":
#     # Example usage
#     target = {'Ag107': 0.5, 'Ag109': 0.5}
#     beamParticle = 'proton'
#     tendl = Tendl(target, beamParticle)
    
#     # Example of retrieving and plotting data
#     tendl.plotTendl23Unique('39', '92')
#     plt.xlabel('Energy (MeV)')
#     plt.ylabel('Cross-section (mb)')
#     plt.title('TENDL-2023 Cross-section Data')
#     plt.legend()
#     plt.show()