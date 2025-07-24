# Paper Questions and Notes
A collection of the important parts of the papers to be used in our own report. Sections marked with "?" means we are unsure if they are relevant or not. 

## Elise
### Q's

### Important Notes for Report 
#### Theory
- 2.1: Selection of radionuclides
    - 2.1.2: Half-life
- 2.2: Production of radionuclides
    - 2.2.1: Cross section 
    - 2.2.2: Gamma decay
    - 2.2.3: Gamma-ray interaction with matter (?)
- 2.3: Positron emission tomography (PET)
- 2.4: Targeted radionuclide therapy
- 2.5: Theranostics (?)

#### The Experiment
- 3.1: The stacked target activation method
- 3.3: Stack design
- 3.5: Gamma-ray spectroscopy
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
    
#### Analysis
- 4.1: Analysis of the spectra
- 4.2: Calculation of end-of-beam activities and production rates
- 4.7: Cross section calculations
    
---

## Hannah 
### Q's

### Important Notes for Report
#### Background and concepts in targeted radionuclide therapy
- 2.1: Targeted radionuclide therapy
- 2.3: Production of radionuclides
- 2.4: Nuclear reactions and reaction cross sections

#### Expermimental setup
- 3.1: The stacked target activation method
$$
σ = \frac{A_0}{N_T Φ(1 - e^{-λΔt_{\text[irr]}})}
$$
with $σ$ being the cross section, $A_0$ the initial activity, $N_T$ being the number of target nuclei, $Φ$ the beam flux and $Δt_{\text[irr]}$ the irradiation time. 
- 3.3: Characterization of the target and monitor
foils
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
    
#### Analysis
- 4.2: Calculation of activities at end of beam
- 4.3: Monitor reactions
- 4.4: Deuteron beam current and energy assignment
$$
Φ(E_d) = \frac{A_0}{N_T σ(E_D)_{\text{mon}}(1-e^{-λΔt_{\text[irr]}})}
$$
with $Φ(E_d)$ being the beam flux at deuteron energy $E_d$, $A_0$ being the end of beam activity, $N_T$ being the number of target nuclei and $σ(E_D)_{\text{mon}}$ being the cross section from the monitor data from IAEA database.
- 4.5: Cross sections (?)