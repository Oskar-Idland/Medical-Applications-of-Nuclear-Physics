# **Terminology**

$A$ — **Activity** ($\mathrm{Bq}=\mathrm{s}^{-1}$): Number of radioactive nuclei that decay per second.

---

$\sigma$ — **Cross section** ($\mathrm{b}=10^{-28} \; \mathrm{m}^2$): Measure of the probability of a nuclear interaction between a particle and a nucleus.

---

$\lambda$ — **Decay constant** ($\mathrm{s}^{-1}$): Probability per unit time that a nucleus will decay.

---

$t_{1/2}$ — **Half-life** ($\mathrm{s}$): Time it takes for half of the radioactive nuclei to decay. Related to decay-life by $t_{1/2} = \ln 2/\lambda$.

---

$N$ — **Number of radioactive nuclei** ($1$): Number of undecayed radioactive nuclei at a given time.

---

$\Phi$ — **Particle flux (beam current)** ($\mathrm{particles}/m²·s$): Number of particles passing through a unit area per second.

---
<br><br><br>



# **Derivation of Cross Section Formula**

When a target foil is irradiated by a particle beam, the production rate $R$ of a nuclear reaction product depends on:

- Number of target nuclei, $N_T$  
- Particle flux (beam current), $\Phi$  
- Reaction cross section, $\sigma$  

<br>

This is expressed by the **thin target approximation**:

$$
R = N_T \Phi \sigma
$$

Here, the beam energy loss in the target is negligible. Units require either a homogeneous target density or a uniform beam intensity.

---

The number of product nuclei $N(t)$ changes over time according to:

$$
\frac{dN}{dt} = R - \lambda N
$$

where $\lambda$ is the decay constant.<br>
Solving with initial condition $N(0) = 0$:

$$
N(t) = \frac{R}{\lambda} \left(1 - e^{-\lambda t}\right)
$$

---

The activity $A(t)$ is:

$$
A(t) = \lambda N(t) = R \left(1 - e^{-\lambda t}\right) = N_T \Phi \sigma \left(1 - e^{-\lambda t}\right)
$$

At the end of irradiation time $\Delta t_{\mathrm{irr}}$, the activity is:

$$
A_0 = N_T \Phi \sigma \left(1 - e^{-\lambda \Delta t_{irr}}\right)
$$

Rearranging to solve for the cross section:

$$
\sigma = \frac{A_0}{N_T \Phi \left(1 - e^{-\lambda \Delta t_{irr}}\right)}
$$

This formula assumes the thin target approximation, constant beam current over time, and either a uniform beam intensity or a homogeneous target density.

---
<br><br><br>

# **Finding the Initial Activity**
We can find the initial activity $A_0$ from the measured gamma-ray spectrum. The only directly observed quantity is the number of counts, $N_C$, which depends on:

* the detector efficiency, $\epsilon$
* the gamma-ray emission intensity, $I_\gamma$
* the decay constant, $\lambda$
* the counting time, $\Delta t_c$
* the delay time before counting, $\Delta t_d$

The expression for the initial activity is:

$$
A_0=\frac{N_C\lambda}{\epsilon I_\gamma (1-e^{-\lambda \Delta t_c})e^{-\lambda \Delta t_d}}
$$

From the experiments, we also have the values for $\Delta t_c$ and $\Delta t_d$, while $\lambda$ can be found in tables for the specific isotope.

To determine $I_\gamma$, we use the NuDat database to look up the specific radionuclide. The gamma-ray intensity is listed as a percentage per 100 decays and is converted to a decimal fraction before being used in the equation.

The detector efficiency $\epsilon$ is determined through an energy-dependent efficiency calibration. This is done by measuring standard sources with well-known gamma-ray energies and intensities. In our case, we used $^{137}\mathrm{Cs}$, $^{133}\mathrm{Ba}$, and $^{152}\mathrm{Eu}$ for calibration. The measured count rates from these sources are used to construct a calibration curve of efficiency versus energy. From this curve, we interpolate the efficiency at the energy of the gamma-ray of interest.

The detector efficiency $\epsilon$ at a specific gamma energy is calculated as:

$$
\epsilon = \frac{N_C}{A  I_\gamma \Delta t_c}
$$

where $A$ is the activity of the source at the measurement time. This formula means the efficiency is the ratio between the detected gamma counts and the expected number of gamma photons emitted during the counting time.

---
<br>

### **Calculating Calibration Source Activities**

To calculate the detector efficiency, we first need the activities of the calibration sources at the time of measurement. We start with the initial activity given on the source certificate and decay it to the measurement date using the radionuclide's known half-life.

The activity at time $t$ is calculated as:

$$
A(t) = A_0e^{-\lambda t},
$$

where $t$ is the elapsed time since the reference date.
<br><br>

---

Measurement date: 11 June 2025, 11:00 (CEST)  
Conversion: $1 \;µ\text{Ci} = 37000 \; \text{Bq}$

---

<br>

#### Ba-137
- Source no.: 2R160  
- $A_0$: 10.78 µCi  
- $t_{1/2}$: 10.54 years  
- Reference date: 1 October 1988  
- Elapsed time: 36.69 years  
- $\lambda = 0.0658 \, \mathrm{year}^{-1}$
- $ A_{\text{Ba}} = 44.444 \, \mathrm{kBq} $

---

#### Cs-137  
- Source no.: 1S134  
- $A_0$: 11.46 µCi  
- $t_{1/2}$: 30.17 years  
- Reference date: 1 February 1978  
- Elapsed time: 47.36 years  
- $\lambda = 0.0230 \, \mathrm{year}^{-1}$  
- $ A_{\text{Cs}} = 147.383 \, \mathrm{kBq} $

---

#### Eu-152  
- $A_0$: 150000 Bq  
- $t_{1/2}$: 13.52 years  
- Reference date: 1 January 2002  
- Elapsed time: 23.45 years  
- $\lambda = 0.0513 \, \mathrm{year}^{-1}$  
- $A_{\text{Eu}} =  33.122 \, \mathrm{kBq}$


---
<br><br><br>

# **Medical Radionuclides**
Medical radionuclides are radioactive substances used in healthcare for diagnosing and treating diseases. In diagnostics, they can be tracked inside the body using special cameras, allowing doctors to see how organs and tissues are functioning. This is common in imaging techniques like PET and SPECT scans. In therapy, radionuclides are used to destroy cancer cells by emitting radiation that directly damages them. Medical radionuclides need to have properties that make them safe and effective, such as an appropriate half-life and the right type of radiation. Their use enables more precise diagnosis and targeted treatment. Because of this, radionuclides play an important role in modern medicine.

---
<br><br>

# **The Stacked Target Activation Method**
### Why Measure Cross Sections

Nuclear cross sections are measured to optimize the production of radionuclides used in medicine — whether for therapy, diagnostics, or theranostics. These radionuclides must have favorable properties, such as an appropriate half-life, suitable decay mode, and ideal emission characteristics.

To meet clinical needs and reduce production cost, it's crucial find efficient production routes. This involves choosing target materials with high natural abundance, favorable chemical properties, and accessible beam parameters. Since cross sections depend on beam energy, measuring them across a range of energies yields a curve that shows where the desired reaction is most likely to occur.

This allows us to select an energy that maximizes the production of the radionuclide of interest while minimizing unwanted byproducts. This is key to producing high-purity medical isotopes in an efficient, reliable, and affordable way.

---

### The Method
The stacked target activation method measures nuclear reaction cross sections at multiple energies by irradiating a stack of thin foils with a charged-particle beam. As the beam passes through the stack, its energy decreases, so each foil experiences a different energy. The stack includes both target foils of interest and monitor foils. Monitor foils have well-known properties and known "monitor reactions" with established cross sections, which help determine the beam energy and current in each layer. The target foils are often less well studied or have inconsistent data, so precise measurements are needed. By analyzing the activation of each foil and using the monitor foils to accurately measure the beam current, this method provides reliable cross section data across a range of energies.


3.5: Gamma-ray spectroscopy
    - 3.5.2: Determination of activity from fitted peaks
    $$
    A(t) = A_0 e^{-λt}
    $$
    with $λ$ being the decay constant and $A_0$ the initial activity. 
    $$
    A(Δt_c) = \frac{N_C λ}{εI_γ(1 - e^{-λΔt_c})}
    $$
    $$
    A_0 = \frac{N_C λ}{εI_γ(1 - e^{-λΔt_c})e^{-λΔt_d}} 
    $$
    with $N_C$ being the number of counts in the peak, $ε$ the efficiency, $I_γ$ the gamma intensity, $Δt_d$ the time between the end of irradiation and the start of counting and $Δt_c$ the counting time. 
    - 3.5.4: Efficiency calibration
    

    - 3.4: Gamma-ray spectroscopy
    - 3.4.2: Determination of activity from fitted peaks
    $$
    A(Δt_c) = \frac{N_C λ}{εI_γ(1 - e^{-λΔt_c})} 
    $$
    $$
    A_0 = \frac{N_C λ}{εI_γ(1 - e^{-λΔt_c})e^{-λΔt_d}} 
    $$
    - 3.4.3: Energy and peak shape calibration
    $$
    E = a + b ⋅ c
    $$
    with $E$ being the energy,a is the slope of the line, b is the intercept and c is the channel number at that energy
    - 3.4.4: Efficiency calibration
    $$
    ε(E_γ) = \frac{N_C}{A_0 I_γ(1 - e^{-λΔt_c})e^{-λΔt_d}}
    $$