from tendl import Tendl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def format_number(n):
    return str(n).zfill(3)

df = pd.read_csv("isotopes_sn_irradiation_filtered.csv")

# first_isotope = df.iloc[68]  # or any index you want
# element = first_isotope["Element"]
# mass_number = int(first_isotope["MassNumber"])

element_to_Z = {
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51
}


# Define your natural Sn target
target = {
    "Sn112": 0.0097, "Sn114": 0.0066, "Sn115": 0.0034,
    "Sn116": 0.1454, "Sn117": 0.0768, "Sn118": 0.2422,
    "Sn119": 0.0859, "Sn120": 0.3258, "Sn122": 0.0463, "Sn124": 0.0579
}


beamParticle = "proton"
tendl = Tendl(target, beamParticle)



results = []


for idx, row in df.iterrows():
    element = row["Element"]
    mass_number = int(row["MassNumber"])

    productZ = element_to_Z.get(element, None)
    if productZ is None:
        print(f"Skipping unknown element {element} at index {idx}")
        continue

    try:
        E, Cs = tendl.tendlData(format_number(productZ), format_number(mass_number))
        if len(E) == 0 or len(Cs) == 0:
            print(f"No data for {element}{mass_number} at index {idx}, skipping.")
            continue

        max_idx = np.argmax(Cs)
        max_cs = float(f"{Cs[max_idx]:.4g}")  # 4 significant digits
        energy_at_max = float(f"{E[max_idx]:.4g}")  # 4 significant digits

        results.append({
            "Index": idx,
            "Element": element,
            "MassNumber": mass_number,
            "MaxCrossSection_mb": max_cs,
            "EnergyAtMax_MeV": energy_at_max
        })

        print(f"{element}{mass_number}: Max XS = {max_cs} mb at {energy_at_max} MeV")


    except Exception as e:
        print(f"Error processing {element}{mass_number} at index {idx}: {e}")
        continue

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("max_cross_sections.csv", index=False)

print("\nResults saved to 'max_cross_sections.csv'.")












# productZ = element_to_Z.get(element, None)
# if productZ is None:
#     raise ValueError(f"Unknown element symbol: {element}")

# target_key = f"{element}{mass_number}"
# target = {target_key: 1.0}
# target = {"Sn112": 0.0097, "Sn114": 0.0066, "Sn115": 0.0034, "Sn116": 0.1454, "Sn117": 0.0768, "Sn118": 0.2422, "Sn119": 0.0859, "Sn120": 0.3258, "Sn122": 0.0463, "Sn124": 0.0579}



# Use padded strings for productZ and productA (Z and A)
# tendl.plotTendl23Unique(format_number(productZ), format_number(mass_number))
# plt.title(f"TENDL data for {element}{mass_number} with proton beam")
# plt.xlabel("Energy (MeV)")
# plt.ylabel("Cross section (mb)")
# plt.show()


# # We're using energies up to 55 MeV