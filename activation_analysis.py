import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import curie as ci
from uncertainties import ufloat
from collections import defaultdict
from scipy.optimize import curve_fit
from datetime import timedelta
from datetime import datetime


plt.rcParams.update({
    "font.size": 14,
    "legend.fontsize": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})
class ActivityAnalyzer:
    def __init__(self, spectra_dir, calibration_path, isotopes, job_filter="job1"):
        self.spectra_dir = Path(spectra_dir)
        self.calibration_path = calibration_path
        self.isotopes = isotopes
        self.job_filter = job_filter

        self.calibration = None
        self.spectra = []
        self.peak_times = defaultdict(lambda: defaultdict(list))
        self.peak_activities = defaultdict(lambda: defaultdict(list))

    def load_calibration(self):
        self.calibration = ci.Calibration(self.calibration_path)

    def load_spectra(self):
        spectra_files = sorted(self.spectra_dir.glob("job*.Spe"))
        jobs = defaultdict(list)
        for file_path in spectra_files:
            job_prefix = file_path.name.split('_')[0]
            jobs[job_prefix].append(file_path)

        filtered_jobs = {k: v for k, v in jobs.items() if k == self.job_filter}

        self.spectra.clear()
        for job_name, files in filtered_jobs.items():
            for file_path in sorted(files):
                sp = ci.Spectrum(str(file_path))
                sp.isotopes = self.isotopes
                sp.cb = self.calibration
                start_time = self.get_spectrum_start_time(file_path)
                self.spectra.append((job_name, sp, start_time))

    def extract_peak_data(self, spectrum):
        peaks = []
        for energy, isotope, counts, counts_unc, intensity, intensity_unc, efficiency, efficiency_unc, real_time, live_time in zip(
            spectrum.peaks["energy"].array,
            spectrum.peaks["isotope"],
            spectrum.peaks["counts"].array,
            spectrum.peaks["unc_counts"].array,
            spectrum.peaks["intensity"].array,
            spectrum.peaks["unc_intensity"].array,
            spectrum.peaks["efficiency"].array,
            spectrum.peaks["unc_efficiency"].array,
            spectrum.peaks["real_time"].array,
            spectrum.peaks["live_time"].array
        ):
            data = {
                "isotope": isotope,
                "counts": counts,
                "counts_unc": counts_unc,
                "intensity": intensity,
                "intensity_unc": intensity_unc,
                "efficiency": efficiency,
                "efficiency_unc": efficiency_unc,
                "real_time": real_time,
                "live_time": live_time,
                "decay_constant": ci.Isotope(isotope).decay_const()
            }
            peaks.append((energy, data))
        return peaks

    def calculate_activity(self, data):
        N_counts_u = ufloat(data["counts"], data["counts_unc"])
        intensity_u = ufloat(data["intensity"], data["intensity_unc"])
        efficiency_u = ufloat(data["efficiency"], data["efficiency_unc"])
        decay_constant = data["decay_constant"]
        live_time = data["live_time"]

        activity = (N_counts_u * decay_constant) / (efficiency_u * intensity_u * (1 - np.exp(-decay_constant * live_time)))
        return activity

    # def analyze(self):
    #     self.peak_times.clear()
    #     self.peak_activities.clear()
    #     cumulative_time = 0

    #     for job_name, spectrum in self.spectra:
    #         spectrum.fit_peaks(skew_fit=True)
    #         peak_data = self.extract_peak_data(spectrum)
    #         if not peak_data:
    #             continue

    #         # Dictionary to keep track of best peak per isotope for this spectrum
    #         best_peaks = {}

    #         # Find peak with max counts for each isotope
    #         for energy, data in peak_data:
    #             isotope = data["isotope"]
    #             counts = data["counts"]
    #             # Update if first or if counts higher than previous
    #             if isotope not in best_peaks or counts > best_peaks[isotope][1]:
    #                 best_peaks[isotope] = (energy, counts, data)

    #         # Debug print best peaks found for this spectrum
    #         #print(f"Best peaks found for {job_name} in {spectrum.filename}:")
    #         # for isotope, (energy, counts, data) in best_peaks.items():
    #         #     if isotope == "108AG":
    #                 #print(f"Isotope {isotope}: Energy = {energy:.2f} keV, Counts = {counts:.2f}")
    #                 #print(f"{counts:.2f}")

    #         measurement_duration = peak_data[0][1]["real_time"]

    #         # Calculate activity and store data only for best peaks
    #         for isotope, (energy, counts, data) in best_peaks.items():
    #             activity = self.calculate_activity(data).nominal_value
    #             self.peak_times[isotope][energy].append(cumulative_time)
    #             self.peak_activities[isotope][energy].append(activity)

    #         cumulative_time += measurement_duration
    
    

    # def plot_red_peak_stacked(self, E_peak, window):      
    #     for i, (job_name, sp) in enumerate(self.spectra):
    #         channels = np.arange(len(sp.hist))
    #         energies = sp.cb.eng(channels)
    #         counts = sp.hist

    #         mask = (energies >= E_peak - window) & (energies <= E_peak + window)
    #         energies_zoom = energies[mask]
    #         counts_zoom = counts[mask]

    #         # Plot histogram bars instead of line plot
    #         plt.bar(energies_zoom, counts_zoom, width=(energies_zoom[1] - energies_zoom[0]), alpha=0.3, label=i)
    #         plt.ylabel(f'Counts\n{i+1}')
    #         if i == 1:
    #             break
    #     plt.xlim([E_peak - window, E_peak + window])
    #     plt.legend()
    #     plt.xlabel('Energy (keV)')
    #     plt.tight_layout()
    #     plt.show()

    def analyze(self, delay_time=7):
        self.peak_times.clear()
        self.peak_activities.clear()

        # Finn earliest start time som EOB
        start_times = [st for _, _, st in self.spectra if st is not None]
        if not start_times:
            raise RuntimeError("No start times found in spectra")
        t0 = min(start_times)

        for job_name, spectrum, start_time in self.spectra:
            spectrum.fit_peaks(skew_fit=True)
            peak_data = self.extract_peak_data(spectrum)
            if not peak_data:
                continue

            best_peaks = {}
            for energy, data in peak_data:
                isotope = data["isotope"]
                counts = data["counts"]
                if isotope not in best_peaks or counts > best_peaks[isotope][1]:
                    best_peaks[isotope] = (energy, counts, data)

            # Seconds after EOB + delay
            elapsed_time = (start_time - t0).total_seconds() + delay_time

            for isotope, (energy, counts, data) in best_peaks.items():
                activity = self.calculate_activity(data)
                self.peak_times[isotope][energy].append(elapsed_time)
                self.peak_activities[isotope][energy].append(activity)




    # def plot_activities(self):
    #     isotopes = sorted(self.peak_times.keys())
    #     fig, axs = plt.subplots(1, len(isotopes), figsize=(6 * len(isotopes), 5), sharey=False)

    #     if len(isotopes) == 1:
    #         axs = [axs]

    #     def decay_model(t, A0, lambd):
    #         return A0 * np.exp(-lambd * t)
        
    #     markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'p']
    #     for ax, isotope in zip(axs, isotopes):
    #         lambd = ci.Isotope(isotope).decay_const()  # decay constant (1/s)

    #         for i, energy in enumerate(sorted(self.peak_times[isotope].keys())):
    #             times = np.array(self.peak_times[isotope][energy])
    #             acts = np.array(self.peak_activities[isotope][energy])

    #             ax.scatter(times, acts, marker=markers[i % len(markers)], label=f'{energy:.2f} keV')

    #             if len(times) == 1:
    #                 # Direct calculation for A0 with single data point
    #                 t1 = times[0]
    #                 A1 = acts[0]
    #                 A0_fit = A1 * np.exp(lambd * t1)
    #                 r_squared = 1.0  # perfect fit by definition

    #                 fit_times = np.linspace(0, t1*1.2, 200)
    #                 fit_activities = A0_fit * np.exp(-lambd * fit_times)

    #                 ax.plot(
    #                     fit_times,
    #                     fit_activities,
    #                     color="#57a3f5",
    #                     label=fr"$A_0 = {A0_fit:.2e}\ \mathrm{{Bq}}$" + f"\n$R^2 = {r_squared:.3f}$"
    #                 )
    #                 print("job4, A0:", A0_fit)

    #             elif len(times) > 1:
    #                 # Fit A0 using curve_fit with fixed lambda
    #                 def decay_model_fixed_lambda(t, A0):
    #                     return A0 * np.exp(-lambd * t)

    #                 try:
    #                     popt, _ = curve_fit(decay_model_fixed_lambda, times, acts, p0=(max(acts),))
    #                     A0_fit = popt[0]

    #                     fit_times = np.linspace(min(times), max(times), 200)
    #                     fit_activities = decay_model_fixed_lambda(fit_times, A0_fit)

    #                     residuals = acts - decay_model_fixed_lambda(times, A0_fit)
    #                     ss_res = np.sum(residuals ** 2)
    #                     ss_tot = np.sum((acts - np.mean(acts)) ** 2)
    #                     r_squared = 1 - (ss_res / ss_tot)

    #                     ax.plot(
    #                         fit_times,
    #                         fit_activities,
    #                         color="#57a3f5",
    #                         label=f"$R^2 = {r_squared:.3f}$"
    #                     )
    #                     print(fr"$A_0 = {A0_fit:.2e}\ \mathrm{{Bq}}$")
    #                 except RuntimeError:
    #                     print(f"Fit failed for {isotope} at {energy:.1f} keV")
    #                     ax.scatter(times, acts, marker='x', s=40,
    #                             label=f"{energy:.1f} keV (fit failed)")

    #             else:
    #                 # No data points, just skip
    #                 continue

    #         isotope_latex = r"$^{108}$Ag" if isotope == "108AG" else r"$^{110}$Ag"
    #         ax.set_title(isotope_latex, fontsize=18)
    #         ax.set_xlabel("Time since EOB (s)")
    #         ax.set_ylabel("Activity (Bq)")
    #         ax.legend(fontsize=14)
    #         ax.grid(True, alpha=0.4)
    #         ax.set_xlim(left=0)

    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.savefig(f"figures/{self.job_filter}_activity_curves.pdf")
    #     plt.close()

    def plot_activities(self):
        isotopes = sorted(self.peak_times.keys())
        fig, axs = plt.subplots(1, len(isotopes), figsize=(6 * len(isotopes), 5), sharey=False)

        if len(isotopes) == 1:
            axs = [axs]

        def decay_model_fixed_lambda(t, A0, lambd):
            return A0 * np.exp(-lambd * t)

        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'p']
        for ax, isotope in zip(axs, isotopes):
            lambd = ci.Isotope(isotope).decay_const()  # decay constant (1/s)

            for i, energy in enumerate(sorted(self.peak_times[isotope].keys())):
                times = np.array(self.peak_times[isotope][energy])
                acts_u = np.array(self.peak_activities[isotope][energy])
                acts_nom = np.array([a.nominal_value for a in acts_u])
                acts_std = np.array([a.std_dev for a in acts_u])

                # Plot with error bars
                ax.errorbar(times, acts_nom, yerr=acts_std, fmt=markers[i % len(markers)],
                            capsize=3, label=f'{energy:.2f} keV')

                if len(times) == 1:
                    # Direct A0 calc for 1 point
                    t1 = times[0]
                    A1 = acts_nom[0]
                    A1_unc = acts_std[0]
                    A0_fit = A1 * np.exp(lambd * t1)
                    A0_unc = A1_unc * np.exp(lambd * t1)
                    r_squared = 1.0

                    fit_times = np.linspace(0, t1 * 1.2, 200)
                    fit_activities = A0_fit * np.exp(-lambd * fit_times)

                    ax.plot(
                        fit_times,
                        fit_activities,
                        color="#57a3f5",
                        label=fr"$A_0 = ({A0_fit:.2e} \pm {A0_unc:.1e})\ \mathrm{{Bq}}$" + f"\n$R^2 = {r_squared:.3f}$"
                    )
                    print(f"A0 (single point) = {A0_fit:.2e} ± {A0_unc:.2e} Bq")

                elif len(times) > 1:
                    try:
                        # Fit using uncertainties
                        popt, pcov = curve_fit(
                            lambda t, A0: decay_model_fixed_lambda(t, A0, lambd),
                            times, acts_nom,
                            sigma=acts_std,
                            absolute_sigma=True,
                            p0=[max(acts_nom)]
                        )
                        A0_fit = popt[0]
                        A0_unc = np.sqrt(np.diag(pcov))[0]

                        fit_times = np.linspace(min(times), max(times), 200)
                        fit_activities = decay_model_fixed_lambda(fit_times, A0_fit, lambd)

                        residuals = acts_nom - decay_model_fixed_lambda(times, A0_fit, lambd)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((acts_nom - np.mean(acts_nom)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)

                        ax.plot(
                            fit_times,
                            fit_activities,
                            color="#57a3f5",
                            label=fr"$A_0 = ({A0_fit:.2e} \pm {A0_unc:.1e})\ \mathrm{{Bq}}$" + f"\n$R^2 = {r_squared:.3f}$"
                        )
                        print(f"A0 = {A0_fit:.2e} ± {A0_unc:.2e} Bq, R² = {r_squared:.3f}")
                    except RuntimeError:
                        print(f"Fit failed for {isotope} at {energy:.1f} keV")
                        ax.scatter(times, acts_nom, marker='x', s=40,
                                label=f"{energy:.1f} keV (fit failed)")
                else:
                    continue

            isotope_latex = r"$^{108}$Ag" if isotope == "108AG" else r"$^{110}$Ag"
            ax.set_title(isotope_latex, fontsize=18)
            ax.set_xlabel("Time since EOB (s)")
            ax.set_ylabel("Activity (Bq)")
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(left=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"figures/{self.job_filter}_activity_curves.pdf")
        plt.close()







    def get_spectrum_start_time(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('$DATE_MEA:'):
                    date_str = next(f).strip()
                    # Format: MM/DD/YYYY HH:MM:SS
                    return datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S')



def test_decaychain_fit():
    from datetime import datetime
    exp_path = Path("spectra/experiment")
    spectra_files = sorted(exp_path.glob("job1_*.Spe"))

    EoB_str = "01/01/2025 12:00:00"
    EoB_dt = datetime.strptime(EoB_str, "%m/%d/%Y %H:%M:%S")

    spectra = []
    for f in spectra_files:
        sp = ci.Spectrum(str(f))
        sp.isotopes = ["108AG", "110AG"]
        print(f"{sp.filename} start_time: {sp.start_time} (Unix ts: {sp.start_time.timestamp()})")
        spectra.append(sp)

    dc = ci.DecayChain("108AG", units="s")
    dc.A0 = {"108AGg": 1e5, "110AGg": 5e4}

    dc.get_counts(spectra, EoB=EoB_dt)  # Pass datetime, ikke string

    print("EoB timestamp:", EoB_dt.timestamp())
    print("Raw _counts start times (relative to EoB, in s):", dc._counts['start'].tolist())

    # Ikke trekk fra EoB_ts her, de er allerede relative tider:
    relative_starts = dc._counts['start'].tolist()
    print("Start times relative (seconds since EoB):", relative_starts)

    isotopes_fit, A0_fit, cov = dc.fit_A0()
    print("Fitted isotopes:", isotopes_fit)
    print("Fitted initial activities (Bq):", A0_fit)
    print("Covariance matrix:\n", cov)

    dc.plot()
    plt.show()








if __name__ == "__main__":
    test_decaychain_fit()



# if __name__ == "__main__":
#     # analyzer = ActivityAnalyzer(
#     #     spectra_dir="spectra/experiment",
#     #     calibration_path="results/efficiency_calibration.json",
#     #     isotopes=["108AG", "110AG"],
#     #     job_filter="job1" 
#     # )

#     # analyzer.load_calibration()
#     # analyzer.load_spectra()
#     # analyzer.plot_red_peak_stacked(E_peak=633, window=20)
#     pass