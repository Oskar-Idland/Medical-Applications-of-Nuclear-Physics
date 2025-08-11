# Summer Project 2025: Medical Applications of Nuclear Physics

This repository contains a comprehensive nuclear physics analysis toolkit for studying proton-induced nuclear reactions on metal foils for medical isotope production. The codebase implements uncertainty propagation, cross-section analysis, and counting time optimization for gamma-ray spectroscopy experiments.

## Overview

The project analyzes proton beam activation of layered target stacks containing gallium (Ga) and tin (Sn) foils with copper (Cu) and nickel (Ni) monitor foils. The analysis determines optimal counting times for detecting produced radioisotopes using high-purity germanium (HPGe) detectors.

## Repository Structure

### Core Analysis Files (`src/`)

- **`analysis.ipynb`** - Main Jupyter notebook containing the complete analysis workflow:
  - Detector calibration and efficiency analysis
  - Activity decay analysis for five experimental datasets
  - Cross-section evaluation for target and monitor materials
  - Stack analysis and counting time calculations with uncertainty propagation

- **`stack_analysis.py`** - Core module for analyzing irradiated target stacks:
  - Calculates isotope activities from cross-sections and irradiation parameters
  - Determines counting times for statistical precision with uncertainty propagation
  - Handles gamma-ray spectroscopy analysis with detector efficiency consideration
  - Provides visualization of counting time dependencies

- **`cs_analysis.py`** - Cross-section analysis module:
  - Analyzes nuclear cross-section data for isotopes produced by particle beam activation
  - Filters isotopes by half-life and cross-section thresholds
  - Handles both natural abundance targets and specific isotope lists
  - Generates overview tables and cross-section plots

- **`spectrum_analysis.py`** - Gamma-ray spectrum analysis:
  - Processes experimental gamma-ray spectra from HPGe detectors
  - Performs activity decay analysis with uncertainty propagation
  - Extracts initial activities and production rates from time-series measurements
  - Handles peak fitting and isotope identification

- **`tendl.py`** - TENDL (Theoretical Evaluation of Nuclear Data Libraries) interface:
  - Provides access to theoretical nuclear cross-section data
  - Handles data retrieval and processing for various isotopes and reactions
  - Supports energy-dependent cross-section calculations

- **`path.py`** - Cross-platform path handling utility for consistent file operations

### Data Directories

#### `spectra/`
Contains experimental gamma-ray spectrum files in `.Spe` and `.Chn` formats:

- **`calibration/`** - Reference spectra from standard sources:
  - `AA110625_Cs137.*` - Cesium-137 calibration source
  - `AB110625_Ba133.*` - Barium-133 calibration source  
  - `AC110625_Eu152.*` - Europium-152 calibration source

- **`experiment/`** - Experimental spectra from five irradiation jobs:
  - `job1_Ag4_1min_real10_loop6_*` - Silver plate #4, 1 min irradiation, 10s measurement, 6 loops
  - `job2_Ag10_2min_real30_loop6_*` - Silver plate #10, 2 min irradiation, 30s measurement, 6 loops
  - `job3_Ag73_3min_real40_loop3_*` - Silver plate #73, 3 min irradiation, 40s measurement, 3 loops
  - `job4_Ag2_3min_real120_loop1_*` - Silver plate #2, 3 min irradiation, 120s measurement, 1 loop
  - `job5_Ag4_3min_real5_loop6_*` - Silver plate #4, 3 min irradiation, 5s measurement, 6 loops

- **`tests/`** - Test and validation spectra

#### `tendl_data/`
Nuclear cross-section data from TENDL database stored as NumPy arrays:

- **`Cu/`** - Cross-section data for copper target reactions (*.npy files for each product isotope)
- **`Ga/`** - Cross-section data for gallium target reactions
- **`Ni/`** - Cross-section data for nickel target reactions  
- **`Sn/`** - Cross-section data for tin target reactions

Each subdirectory contains `.npy` files named by isotope (e.g., `67Ga.npy`, `108Ag.npy`) with energy-dependent cross-section data.

#### `figs/`
Generated analysis figures and plots:

- **`activity_analysis/`** - Activity decay plots and production rate analysis for each experimental job
- **`cross_section_analysis/`** - Cross-section plots for filtered isotopes by target material

### Configuration Files

- **`calibration.json`** - Detector calibration parameters (energy, efficiency, resolution)

## Key Features

- **Uncertainty Propagation**: Complete uncertainty analysis using the `uncertainties` library

- **Cross-Section Analysis**: Comprehensive filtering and analysis of nuclear reaction cross-sections with TENDL database integration

- **Stack Modeling**: Detailed modeling of layered target stacks with energy degradation and material interaction calculations

- **Detector Calibration**: HPGe detector efficiency calibration using multiple reference sources with confidence band analysis

- **Statistical Analysis**: Counting time optimization for achieving desired statistical precision in gamma-ray measurements

## Usage

The main analysis workflow is contained in `analysis.ipynb`. Run the notebook cells sequentially to:

1. Perform detector calibration and efficiency analysis
2. Analyze experimental activity decay data  
3. Evaluate cross-sections for target and monitor materials
4. Calculate optimal counting times for isotope detection
