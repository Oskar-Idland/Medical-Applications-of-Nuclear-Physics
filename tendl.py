import numpy as np
import requests
import matplotlib.pyplot as plt
from urllib.request import urlopen
from numpy.typing import NDArray, ArrayLike
from numpy import float64, int64

class Tendl:
    def __init__(self, target: dict[str, float], beamParticle: str) -> None:
        """
        Initializes the Tendl class with a target dictionary and beam particle type.

        Args:
            target (dict[str, float]): A dictionary where keys are isotope names (e.g., 'Ir191') and values are their respective intensities in fractions (e.g., 0.373).
            beamParticle (str): The type of beam particle ('deuteron', 'proton', or 'alpha').
        """
        # target = {"Ir191": 0.373, "Ir193": 0.627}
        self.target = {self._name_trans_curie_tendl(k): v for k, v in target.items()}  # Convert keys to TENDL format
        self.beamParticle = beamParticle

    def _name_trans_curie_tendl(self, name: str) -> str:
        """
        Converts isotope names from the format '108AG' to 'Ag108'.
        """
        numbers_part = ''.join(filter(str.isdigit, name))
        # Extract the letters and capitalize (e.g., 'AG' -> 'Ag')
        letters_part = ''.join(filter(str.isalpha, name)).capitalize()
        return f"{letters_part}{numbers_part}"

    def _tendlDeuteronData(self, productZ: str, productA: str, isomerLevel: str | None = None) -> tuple[list[NDArray], list[NDArray]]:
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
            print("TENDL: No data found for target: " + targetFoil + " for productZ" + productZ + "and product A: " + productA)
            raise Exception

        # CsSummed = sum(np.concatenate(Cs))
        # E = E[0]
        E, Cs = zip(*(Tools().interpolate(x=E_i, y=Cs_i) for E_i, Cs_i in zip(E, Cs)))

        return list(E), list(Cs)

    def tendlData(self, productZ: str, productA: str, isomerLevel: str | None = None, Elimit: float | None = None) -> tuple[list[NDArray], list[NDArray]]:
        targetFoil = list(self.target.keys())[0][0:2]
        product = self._product(productZ, productA)
        fileEnding = self._tendlFileEnding(isomerLevel)
        E = []
        Cs = []
        for t in self.target.keys():
            data = self._retrieveTendlDataFromUrl(
                self._tendlUrl(targetFoil, t, product, fileEnding), t
            )
            if isinstance(data[0], NDArray) and len(data[0]) > 0 and len(data[1]) > 0:
                E.append(data[0])
                Cs.append(data[1])

        if len(E) == 0 or len(Cs) == 0:
            print("TENDL: No data found for target: " + targetFoil + " for productZ" + productZ + "and product A: " + productA)
            return Exception

        # CsSummed = sum(Cs)
        # E = E[0]
        # x, y, xlimit=None, zeroPadding=False
        # print(Elimit)

        E, Cs = zip(*(Tools().interpolate(x=E_i, y=Cs_i, xlimit=Elimit) for E_i, Cs_i in zip(E, Cs)))

        return list(E), list(Cs)

    def plotTendl23(self, productZ: str, productA: str, isomerLevel: str | None = None) -> None:  # , feeding = None, branchingRatio = None, parentIsomerLevel = None):
        # try:
        E, Cs = self._tendlDeuteronData(productZ, productA, isomerLevel)
        # if feeding == 'beta+' or feeding == 'beta-':
        # CsParent = self.correctForFeeding(productZ, productA, feeding, branchingRatio, parentIsomerLevel)[1]
        # Cs = Cs + CsParent
        plt.plot(E, Cs, label='TENDL-2023', linestyle='--', color='blue')

    # except:
    # print("Unable to retrive tendl data, perhaps no internet connection?")

    def plotTendl23Unique(self, productZ: str, productA: str, Elimit: float | None = None, isomerLevel: str | None = None, color: str = 'blue', lineStyle: str = '--', label: str = 'TENDL-2023') -> None:
        try:
            E, Cs = self.tendlData(productZ, productA, isomerLevel, Elimit)
            plt.plot(E, Cs, label=label, linestyle=lineStyle, color=color)
        except:
            print("Unable to retrive tendl data, perhaps no internet connection?")

    def plotdataWithMultipleFeeding(self, productZ: str, productA: str, isomerLevel: str, betaPlusDecayChain: dict[str, list[str | float]] | None = None, betaMinusDecayChain: dict[str, list[str | float]] | None = None, isomerDecayChain: dict[str, list[str | float]] | None = None) -> None:
        # {isotope: [productZ, branchingRatio isomerLevel]} #beta+/beta-
        # {isotope: [branchingRatio isomerLevel]} #isomer
        try:
            E, Cs = self._tendlDeuteronData(productZ, productA, isomerLevel)
            Cs_betaplus = []
            Cs_betaminus = []
            Cs_isomer = []

            if betaPlusDecayChain:
                for i in list(betaPlusDecayChain.keys()):
                    Z = betaPlusDecayChain[i][0]
                    branchingRatio = betaPlusDecayChain[i][1]
                    isomerLevel = betaPlusDecayChain[i][2]
                    E_bp, Cs_bp = self._tendlDeuteronData(Z, productA, isomerLevel)
                    Cs_betaplus.append(Cs_bp * branchingRatio)

            if betaMinusDecayChain:
                for i in list(betaMinusDecayChain.keys()):
                    Z = betaMinusDecayChain[i][0]
                    branchingRatio = betaMinusDecayChain[i][1]
                    isomerLevel = betaMinusDecayChain[i][2]
                    E_bm, Cs_bm = self._tendlDeuteronData(Z, productA, isomerLevel)
                    Cs_betaminus.append(Cs_bm * branchingRatio)

            if isomerDecayChain:
                for i in list(isomerDecayChain.keys()):
                    branchingRatio = isomerDecayChain[i][0]
                    isomerLevel = isomerDecayChain[i][1]
                    E_i, Cs_i = self._tendlDeuteronData(productZ, productA, isomerLevel)
                    Cs_isomer.append(Cs_i * branchingRatio)

            totCs = Cs + sum(Cs_betaplus) + sum(Cs_betaminus) + sum(Cs_isomer)
            plt.plot(E, totCs, label='TENDL-2023', linestyle='--', color='blue')

        except:
            print("Unable to retrive tendl data, perhaps no internet connection?")

    def _product(self, productZ: str, productA: str) -> str:
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
        if len(target) < 5:
            target = target[0:2] + '0' + target[2:]

        return (
            'https://tendl.web.psi.ch/tendl_2023/deuteron_file/'
            + targetFoil + '/' + target
            + '/tables/residual/rp'
            + product + fileEnding
        )

    def _tendlUrl(self, targetFoil, target, product, fileEnding):
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
        # Cu65 --> Cu065. Ir193=Ir193
        isotopeNumber = targetIsotope[len(targetFoil):]
        formattedIsotopeNumber = isotopeNumber if len(isotopeNumber) == 3 else '0' + isotopeNumber

        return targetFoil + formattedIsotopeNumber

    def _tendlFileEnding(self, isomerLevel: str | None = None) -> str:
        return '.tot' if isomerLevel is None else '.L' + isomerLevel

    def _retrieveTendlDataFromUrl(self, url: str, target: str) -> tuple[NDArray, NDArray]:
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
        # tendl_data = np.genfromtxt(urlopen('ttps://tendl.web.psi.ch/tendl_2023/deuteron_file/Ir/Ir193/tables/residual/rp078193.L05', delimiter=" "))
        tendl_data = np.genfromtxt(urlopen(url, delimiter=" "))
        energy = tendl_data[:, 0]
        xs = tendl_data[:, 1]

        return energy, xs

from scipy.interpolate import splev, splrep
import numpy as np

class Tools:
    def interpolate(self, x: ArrayLike, y: ArrayLike, xlimit: float | None = None, zeroPadding: bool = False) -> tuple[ArrayLike, ArrayLike]:
        if zeroPadding:
            x, y = self.zeroPadding(x, y)
        tck = splrep(x, y, s=0, k=5)
        if xlimit:
            x = np.linspace(1, xlimit, 1000)
        # else:
        #     x_new = np.linspace(1, 100, 1000)

        y = splev(x, tck, der=0)
        return x, y

    def zeroPadding(self, x: NDArray[float64 | int64], y: NDArray[float64 | int64]) -> tuple[NDArray[float64 | int64], NDArray[float64 | int64]]:
        if x[0] != 0:
            zero_padding = np.linspace(0, x[0] - 0.5, 10)
            zeros_y = np.zeros((len(zero_padding)))
            x = np.concatenate((zero_padding, x))
            y = np.concatenate((zeros_y, y))

        return x, y

if __name__ == "__main__":
    # Example usage
    target = {"Ga69": 0.373, "Ga71": 0.627}
    beamParticle = "proton"
    tendl = Tendl(target, beamParticle)
    E, Cs = tendl.tendlData("32", "69")
    print(E)
