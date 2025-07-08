import pandas as pd

df = pd.read_csv("isotopes_sn_irradiation.csv")


time_units = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 86400,
    "y": 31557600,
}

def parse_half_life_to_sec(hl_string):
    try:
        parts = hl_string.lower().split()
        value = float(parts[0].replace("×10", "e"))  # Handle scientific notation
        unit = parts[1]
        multiplier = time_units.get(unit, None)
        if multiplier is None:
            return None
        return value * multiplier
    except:
        return None
    

# Apply parsing to get half-life in seconds
df["HalfLife_s"] = df["HalfLife"].apply(parse_half_life_to_sec)

# Filter out < 60 sec and > 100 years (3155760000 sec)
df_filtered = df[(df["HalfLife_s"] >= 60) & (df["HalfLife_s"] <= 3155760000)]

# Drop the parsed helper column to only store the original columns
df_filtered = df_filtered.drop(columns=["HalfLife_s"])

# Save to new CSV
df_filtered.to_csv("isotopes_sn_irradiation_filtered.csv", index=False)

# Print out the number of isotopes after filtering
print(f"Number of isotopes after filtering: {len(df_filtered)}")