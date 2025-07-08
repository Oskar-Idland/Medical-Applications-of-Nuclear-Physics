import pandas as pd

# Load CSV
df = pd.read_csv("max_cross_sections.csv")

# Convert MaxCrossSection_mb to numeric if it was saved as string
df["MaxCrossSection_mb"] = pd.to_numeric(df["MaxCrossSection_mb"], errors="coerce")

# Drop any rows with NaNs in MaxCrossSection_mb
df = df.dropna(subset=["MaxCrossSection_mb"])

# Sort by MaxCrossSection_mb ascending
df_sorted = df.sort_values(by="MaxCrossSection_mb", ascending=True)

# Select the 20 lowest
lowest_20 = df_sorted.head(20)

# Print formatted output
print(f"{'Element':<8} {'MassNumber':<12} {'Max_CS [mb]':<12} {'Energy [MeV]':<12}")
print("-" * 50)
for idx, row in lowest_20.iterrows():
    print(f"    {row['Element']:<8} {int(row['MassNumber']):<12} {row['MaxCrossSection_mb']:<12} {row['EnergyAtMax_MeV']:<12}")
