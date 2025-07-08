import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tendl import Tendl

class IsotopeFinder:
    def __init__(self, isotope_csv, target, beam_particle="proton"):
        self.df = pd.read_csv(isotope_csv)
        self.target = target
        self.beam_particle = beam_particle
        self.tendl = Tendl(self.target, self.beam_particle)

    def half_life_to_seconds(self, hl_string):
        time_units = {"ms": 1e-3, "s": 1, "min": 60, "h": 3600, "d": 86400, "y": 31557600}
        value_str, unit = hl_string.split()
        return float(value_str) * time_units.get(unit, np.nan)

    

# What i want to do:


"""
1. From the original list of all isotopes, i want to filter out all the stable ones, and other
   with too long or too short half-lives for example. And then write this list to a csv file.

2. Then, I want to use Tendl to iterate through the first filtered list and creating a new list
   of potentially viewved isotopes with: 133Cd  Half-life   Max_CS value (at x MeV), where the ones
   with very low CS_values are filtered out.

3. I also want to plot all the filtered out isotopes with tendl to double check that they should be
   filtered out.

4. Make a function to plot one isotope with Tendl
5. Make a function to return out maximum CS value with an energy limit
"""