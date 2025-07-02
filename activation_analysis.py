import curie as ci
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define base folder for spectra files
base_folder = Path("../spectra/experiment")
def generate_files(base_pattern, count):
    """Generate a list of file paths based on a base pattern and count."""
    return [base_folder / f"{base_pattern}{i}.Spe" for i in range(count)]


# Load the Calibration object from a JSON file
calibration_file = "detector_efficiency_calibration.json"
calibration = ci.Calibration(calibration_file)


jobs = [
    {"disk_id": 4, "t_irr": 60, "Δt_d": 15, "Δt_c": 10, "files": generate_files("job1_Ag4_1min_real10_loop600", 6)},
    {"disk_id": 10, "t_irr": 120, "Δt_d": 11, "Δt_c": 30, "files": generate_files("job2_Ag10_2min_real30_loop600", 6)},
    {"disk_id": 73, "t_irr": 180, "Δt_d": 11, "Δt_c": 40, "files": generate_files("job3_Ag73_3min_real40_loop3_00", 3)},
    {"disk_id": 2, "t_irr": 180, "Δt_d": 14, "Δt_c": 120, "files": generate_files("job4_Ag2_3min_real120_loop1_00", 1)},
    {"disk_id": 4, "t_irr": 180, "Δt_d": 7, "Δt_c": 5, "files": generate_files("job5_Ag5_3min_real5_loop6_00", 6)}
]


# Target peaks (half-life in seconds)
target_peaks = {
    "108AG": [
        {"energy": 632.98, "intensity": 0.0176, "unc_intensity": 0.001, "half_life": 2.382 * 60},
        {"energy": 433.96, "intensity": 0.0050, "unc_intensity": 0.001, "half_life": 2.382 * 60},  # missing uncertainty
        {"energy": 618.86, "intensity": 0.00261, "unc_intensity": 0.00022, "half_life": 2.382 * 60},
    ],
    "110AG": [
        {"energy": 657.50, "intensity": 0.0450, "unc_intensity": 0.001, "half_life": 24.56},  # missing uncertainty
    ],
}



if __name__ == "__main__":
    def load_job_spectra(job):
        spectra = [ci.Spectrum(str(f)) for f in job["files"]]
        return spectra

    for job in jobs:
        spectra = load_job_spectra(job)
        for spe in spectra:
            spe.cb = calibration
            spe.isotopes = ["108AG", "110AG"]  # Set isotopes for the spectrum
            spe.plot()
    #         spe.saveas("peak_data.csv")
    #     job["spectra"] = spectra

    # def analyze_job_activity(job, target_peaks):
    #     spectra = job["spectra"]
    #     t_irr = job["t_irr"]
    #     Δt_d = job["Δt_d"]
    #     Δt_c = job["Δt_c"]
        
    #     fig, ax = plt.subplots()
        
    #     # For each isotope and its peaks
    #     for isotope, peaks in target_peaks.items():
    #         for peak in peaks:
    #             energies = []
    #             activities = []
    #             for spe in spectra:
    #                 channel_guess = calibration.map_channel(peak["energy"])
    #                 spe.auto_calibrate(peaks=[[channel_guess, peak["energy"]]])
                    
    #                 # Fit peak and get area
    #                 gammas_df = pd.DataFrame([{
    #                     "isotope": isotope,
    #                     **peak
    #                 }])
    #                 fit_result = spe.fit_peaks(gammas=gammas_df, bg="snip", SNR_min=3, multi_max=2, ident_idx=0)

                    
    #                 if fit_result is not None and not fit_result.empty:
    #                     peak_area = fit_result["counts"].iloc[0]
                        
    #                     decay_const = np.log(2) / peak["half_life"]
    #                     I_gamma = peak["intensity"]
    #                     Δt_c = job["Δt_c"]  
    #                     Δt_d = job["Δt_d"]  
    #                     counts = peak_area
    #                     A_start_count = (counts * decay_const) / (I_gamma * (1 - np.exp(-decay_const * Δt_c)))

    #                     A_end_irradiation = A_start_count * np.exp(decay_const * Δt_d)
                        
    #                     energies.append(peak["energy"])
    #                     activities.append(A_end_irradiation)
                
    #             if energies and activities:
    #                 ax.plot(energies, activities, 'o-', label=f"{isotope} {peak['energy']} keV")
        
    #     ax.set_xlabel("Energy (keV)")
    #     ax.set_ylabel("Activity (corrected counts)")
    #     ax.set_title(f"Activity plot for disk {job['disk_id']}")
    #     ax.legend()
    #     plt.show()

    def analyze_job_activity(job, target_peaks):
        spectra = job["spectra"]
        t_irr = job["t_irr"]
        Δt_d = job["Δt_d"]
        Δt_c = job["Δt_c"]
        
        fig, ax = plt.subplots()
        
        # Prepare a dictionary to store time and activity lists per peak
        peak_data = {}  # key: (isotope, energy), value: {"times": [], "activities": []}
        
        for isotope, peaks in target_peaks.items():
            for peak in peaks:
                peak_key = (isotope, peak["energy"])
                peak_data[peak_key] = {"times": [], "activities": []}
        
        for spe in spectra:
            # Get the spectrum start time as datetime object (if possible)
            try:
                time = pd.to_datetime(spe.start_time)
            except Exception:
                # Fallback to real_time or something else numeric if datetime not available
                time = spe.real_time
            
            for isotope, peaks in target_peaks.items():
                for peak in peaks:
                    channel_guess = calibration.map_channel(peak["energy"])
                    spe.auto_calibrate(peaks=[[channel_guess, peak["energy"]]])
                    
                    gammas_df = pd.DataFrame([{
                        "isotope": isotope,
                        **peak
                    }])
                    fit_result = spe.fit_peaks(gammas=gammas_df, bg="snip", SNR_min=3, multi_max=2, ident_idx=0)
                    
                    if fit_result is not None and not fit_result.empty:
                        counts = fit_result["counts"].iloc[0]
                        decay_const = np.log(2) / peak["half_life"]
                        I_gamma = peak["intensity"] / 100  # convert percent to fraction
                        A_start_count = (counts * decay_const) / (I_gamma * (1 - np.exp(-decay_const * Δt_c)))
                        A_end_irradiation = A_start_count * np.exp(decay_const * Δt_d)
                        
                        peak_key = (isotope, peak["energy"])
                        peak_data[peak_key]["times"].append(time)
                        peak_data[peak_key]["activities"].append(A_end_irradiation)
        
        # Plot all peaks on the same plot, unique color and label for each peak
        for (isotope, energy), data in peak_data.items():
            if data["times"] and data["activities"]:
                # Sort data by time for cleaner lines
                sorted_data = sorted(zip(data["times"], data["activities"]))
                times_sorted, activities_sorted = zip(*sorted_data)
                
                ax.plot(times_sorted, activities_sorted, marker='o', label=f"{isotope} {energy} keV")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Activity (corrected counts)")
        ax.set_title(f"Activity vs Time for disk {job['disk_id']}")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

    for job in jobs:
        analyze_job_activity(job, target_peaks)
        break