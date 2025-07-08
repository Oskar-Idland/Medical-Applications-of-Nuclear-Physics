import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable

from tendl import Tendl

class IsotopeFinder:
   def __init__(self, isotope_csv, beam_particle="proton"):
      self.df = pd.read_csv(isotope_csv)
      self.target = {
         "Sn112": 0.0097, "Sn114": 0.0066, "Sn115": 0.0034,
         "Sn116": 0.1454, "Sn117": 0.0768, "Sn118": 0.2422,
         "Sn119": 0.0859, "Sn120": 0.3258, "Sn122": 0.0463, "Sn124": 0.0579
      }
      self.beam_particle = beam_particle
      self.tendl = Tendl(self.target, self.beam_particle)


   def half_life_to_seconds(self, hl_string):
      time_units = {"ms": 1e-3, "s": 1, "min": 60, "h": 3600, "d": 86400, "y": 31557600}
      if hl_string == "Stable":
         return np.nan
      value_str, unit = hl_string.split()
      return float(value_str) * time_units.get(unit, np.nan)
    

   def filter_by_half_life_range(self, min_hl="60 s", max_hl="1e2 y", filename=None, write_to_file=True):
      min_hl_s = self.half_life_to_seconds(min_hl)
      max_hl_s = self.half_life_to_seconds(max_hl)
      
      self.df["half_life_s"] = self.df["half_life"].apply(self.half_life_to_seconds)
      self.df = self.df[(self.df["half_life_s"] >= min_hl_s) & (self.df["half_life_s"] <= max_hl_s)].reset_index(drop=True)

      
      if write_to_file:
         if filename is None:
            filename = f"filtered_isotopes_{min_hl.replace(' ', '')}_to_{max_hl.replace(' ', '')}.csv"
         self.df.to_csv(filename, index=False)
         print(f"Filtered isotope list saved to '{filename}'")
      else:
         print(f"Filtered isotopes to {len(self.df)} with half-life range ({min_hl} to {max_hl}).")
         return self.df
   

   def get_max_cs_for_isotope(self, element, mass_number, energy_limit=60):
      try:
         Z = periodictable.elements.symbol(element).number
      except Exception:
         print(f"Unknown element symbol: {element}")
         return

      E, Cs = self.tendl.tendlData(str(Z), str(mass_number), Elimit=energy_limit)

      max_idx = np.argmax(Cs)
      max_Cs = float(f"{Cs[max_idx]:.4g}")
      energy_at_max = float(f"{E[max_idx]:.4g}")

      return max_Cs, energy_at_max
   

   def filter_by_min_cs(self, min_Cs=1e-3, filename=None, write_to_file=True):
      filtered_rows = []
      for _, row in self.df.iterrows():
         element = row["element"]
         mass_number = int(row["mass_number"])

         max_Cs, energy_at_max = self.get_max_cs_for_isotope(element, mass_number)
         if max_Cs is None:
            continue

         if max_Cs >= min_Cs:
            row_copy = row.copy()
            row_copy["max_cs"] = max_Cs
            row_copy["energy_at_max"] = f"{energy_at_max} MeV"
            filtered_rows.append(row_copy)
      
      self.df = pd.DataFrame(filtered_rows).reset_index(drop=True)
      
      if write_to_file:
         if filename is None:
            filename = f"filtered_isotopes_min_cs_{min_Cs}.csv"
         self.df.to_csv(filename, index=False)
         print(f"Filtered isotopes to {len(self.df)} with Max. Cs >= {min_Cs} mb.")
         return
   
      return self.df


   def plot_tendl_cs(self, element, mass_number, energy_limit=60):
      try:
         Z = periodictable.elements.symbol(element).number
      except Exception:
         print(f"Unknown element symbol: {element}")
         return
   
      E, Cs = self.tendl.tendlData(str(Z), str(mass_number), E_limit=energy_limit)
      
      plt.plot(E, Cs)
      plt.title(f"TENDL Cross Section for {element}{mass_number}")
      plt.xlabel("Energy (MeV)")
      plt.ylabel("Cross Section (mb)")
      plt.show()
   

   def plot_filtered_out_by_min_cs(self, min_Cs=1e-3, energy_limit=60):
      for _, row in self.df.iterrows():
         element = row["element"]
         mass_number = int(row["mass_number"])

         max_Cs, _ = self.get_max_cs_for_isotope(element, mass_number)
         if max_Cs is None:
            continue

         if max_Cs < min_Cs:
            print(f"Plotting {element}{mass_number} with Max Cs {max_Cs:.4g} mb (which is below the threshold)")
            self.plot_tendl_cs(element, mass_number, energy_limit=energy_limit)


if __name__ == "__main__":
   isotopes = IsotopeFinder("isotopes_sn_irradiation.csv")
   isotopes.filter_by_half_life_range()
   isotopes.filter_by_min_cs()
   # plot_tendl_cs(self, element, mass_number, energy_limit=60):
   # plot_filtered_out_by_min_cs(self, min_Cs=1e-4, energy_limit=60):
   df = pd.read_csv("filtered_isotopes_min_cs_0.001.csv")
   print(len(df))
   for idx, row in df.iterrows():
        print(
            f"{row['element']}{int(row['mass_number'])} | "
            f"{row['half_life']:8} | "
            f"{row['max_cs']} mb"
        )

   
