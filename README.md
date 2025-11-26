# ORCA-VFD  
Open Reliability & Condition Assessment for Variable Frequency Drives

[![DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.30727865-blue.svg)](https://doi.org/10.6084/m9.figshare.30727865)

**Full ORCA-VFD dataset (Figshare DOI):**  
https://doi.org/10.6084/m9.figshare.30727865

ORCA-VFD is an open, field-validated framework for reliability modeling,
remaining useful life (RUL) prediction, anomaly scoring, and degradation
analysis of VFD-driven motor systems. The project integrates physics-based
simulation data, fault-injection data, and multi-year industrial field
measurements into a unified architecture designed for predictive
maintenance research and real-world engineering applications.

This repository accompanies the ORCA-VFD preprint and includes code,
sample datasets, feature engineering tools, model pipelines, and the
standardized ORCA-VFD data format.

---

## Features

- **ORCA-VFD Data Format**  
  Standardized structure for storing physics, fault, and field operating
  data for VFD systems.

- **Lifecycle Synthesis Tools**  
  Generate 100,000-hour degradation trajectories spanning
  infant-mortality, useful-life, and wearout phases.

- **Physics-Informed Feature Engineering**  
  Core-8 VFD reliability features with anomaly normalization and
  deterministic degradation modeling.

- **Remaining Useful Life Models**  
  Random Forest regressors with uncertainty quantification and ensemble
  spread analysis.

- **Economic Optimization Engine**  
  Converts probabilistic RUL predictions into cost-optimal maintenance
  decisions (planned vs unplanned failure cost).

- **Cross-Domain Validation**  
  Maximum Mean Discrepancy (MMD) for measuring distribution shift across
  physics, fault, and field domains.

---

## Repository Structure

ORCA-VFD/
│
├── code/
│ ├── preprocessing/
│ ├── modeling/
│ └── utils/
│
├── data/
│ ├── sample/ # Small sample data for demonstration
│ └── README.md # Instructions for full dataset access
│
├── paper/
│ ├── main.tex # LaTeX source for the ORCA-VFD paper
│ └── figures/
│
├── notebooks/
│ ├── exploration.ipynb
│ └── model_training.ipynb
│
├── LICENSE
└── README.md
