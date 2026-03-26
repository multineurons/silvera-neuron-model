# Silvera Neuron Model

**Copyright (C) 2026 Paolo Giovanni Silvera**  
**License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**  
> Free for academic and research use. Commercial use requires explicit written permission from the author.

> **DISCLAIMER:** This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the author be liable for any claim, damages, or other liability arising from, out of, or in connection with the software or the use thereof. This model is intended for research and educational purposes only. Results should be validated against experimental data before any clinical or commercial application.

---

**Author:** Paolo Giovanni Silvera  
**ORCID:** [0009-0005-9234-1818](https://orcid.org/0009-0005-9234-1818)  
**DOI:** [10.5281/zenodo.19011537](https://doi.org/10.5281/zenodo.19011537)  
**Email:** psilvera@libero.it  
**Year:** 2026  

---

## Overview

The Silvera Model is a single-compartment, continuous dynamical system for simulating cortical spiking neurons. It is governed by **five ordinary differential equations** and **17 tunable parameters**, and requires only **two exponential evaluations** per integration step.

Unlike reset-based models (e.g., Izhikevich, AdEx), the Silvera Model generates action potentials through a **continuous, smooth limit cycle** derived from biophysical gating principles. The state space is fully differentiable at all times, including during and after the spike — a property critical for gradient-based learning and analogue neuromorphic hardware.

---

## Key Features

- **5 state variables:** V (membrane potential), m (Na⁺ activation), n (K⁺ activation), h (Na⁺ inactivation), u (slow adaptation)
- **2 exponential evaluations per step** (for m∞ and n∞ via sigmoid)
- **No discontinuous reset** — spikes arise and terminate through continuous dynamics
- **17 biophysically interpretable parameters**, organised into four functional groups
- **Compatible with gradient-based training** (BPTT, surrogate gradient methods)
- **Suitable for FPGA and analogue neuromorphic implementations**

---

## Key Innovation: Mechanistic h–m Coupling

In Hodgkin–Huxley, the inactivation variable *h* is a direct function of voltage. In the Silvera Model, *h* is driven mechanistically by *m*:

```
dh/dt = (1 - h) · (m · rate_h1 - h · rate_h2)
```

During depolarisation, rising *m* drives *h* upward (inactivation). During recovery, the term −h·rate_h2 returns *h* to zero with time constant τ = 1/rate_h2. This produces an emergent sigmoid recovery trajectory without any precalculated lookup table or voltage-dependent gating function.

---

## Governing Equations

```
Cm · dV/dt = -gL(V - Vrest) - gNa · m · (1-h) · (V - ENa) - gK · n · (V - EK) - u + Istim
dm/dt      = (m∞(V) - m) · rate_Na
dn/dt      = (n∞(V) - n) · rate_K
dh/dt      = (1 - h) · (m · rate_h1 - h · rate_h2)
du/dt      = (g_adapt · (V - Vrest) - u) · rate_slow
```

where the sigmoid activation function is:

```
σ(V; θ, p) = 1 / (1 + exp(-(V - θ) · p))
```

such that m∞(V) = σ(V; th_Na, p_Na) and n∞(V) = σ(V; th_K, p_K).

---

## Validated Phenotypes

The model reproduces the following cardinal cortical firing patterns by varying parameters within the same equation set:

| Phenotype                       | Cell type             | Key features                                               |
|:--------------------------------|:----------------------|:-----------------------------------------------------------|
| **RS** — Regular Spiking        | L2/3 pyramidal        | Tonic firing, spike-frequency adaptation, Class 1 (SNIC)  |
| **FS** — Fast Spiking           | PV+ interneuron       | High-frequency non-adapting firing, Class 2 (Hopf)        |
| **IB** — Intrinsic Bursting     | L5 pyramidal          | Initial burst followed by tonic spiking                    |
| **CH** — Chattering             | L4/6 neurons          | Rhythmic high-frequency burst firing                       |
| **LTS** — Low-Threshold Spiking | SST+ interneuron      | Low threshold, wide spikes, rebound burst                  |
| **FA** — Frequency Adaptation   | L2/3 pyramidal        | Progressive ISI lengthening under sustained stimulation    |
| **RZ** — Resonator              | Cortical interneurons | Subthreshold oscillations (~13 Hz), frequency-selective spiking |
| **NT** — Integrator             | Tonic cortical cells  | Graded, non-oscillatory subthreshold integration           |
| **Rebound Spike**               | Various               | Single spike following hyperpolarising pulse               |
| **Rebound Burst**               | L5 pyramidal / LTS    | Burst following hyperpolarising pulse                      |

### Emergent phenomena (no explicit rules required)
- **Depolarisation block** — natural consequence of h-channel saturation at high current
- **Post-inhibitory rebound spike** — single spike after hyperpolarisation (rate_h2/rate_h1 ≈ 0.26)
- **Post-inhibitory rebound burst** — burst after hyperpolarisation (rate_h2/rate_h1 ≈ 0.67)
- **Class 1 vs Class 2 excitability** — SNIC vs Hopf bifurcation, controlled by th_K

---

## Repository Contents

| File                    | Description                                                        |
|:------------------------|:-------------------------------------------------------------------|
| `model_silvera.sci`     | Reference implementation in Scilab (RK4)                          |
| `model_silvera.py`      | Python implementation (NumPy, RK4)                                 |
| `plot_phenotypes.py`    | Generates publication-quality voltage trace figure (PDF)           |
| `phenotype_traces.pdf`  | Figure: voltage traces and spike overlays for all phenotypes       |
| `*.xml`                 | Parameter files for each validated phenotype                       |
| `CITATION.cff`          | Citation metadata (CFF format)                                     |
| `LICENSE`               | License file (CC BY-NC-SA 4.0)                                     |
| `README.md`             | This file                                                          |

---

## Scilab Usage

Requires **Scilab 6.x or later** (free, open-source). No additional toolboxes required.

```scilab
// Open Scilab, then run:
exec('model_silvera.sci', -1)
```

Parameters can be modified directly in the script. Phenotype-specific parameter sets are provided in the accompanying XML files.

---

## Python Usage

### Dependencies

```
python     >= 3.8
numpy      >= 1.21
matplotlib >= 3.4
```

Install with pip:

```bash
pip install numpy matplotlib
```

### Run the model

```bash
python model_silvera.py
```

Parameters can be modified directly at the top of the script (section 1. PARAMETERS).

### Generate the phenotype figure

Place `plot_phenotypes.py` in the same directory as the XML parameter files, then run:

```bash
python plot_phenotypes.py
```

This will simulate all phenotypes found in the directory and save `phenotype_traces.pdf`.

---

## Parameter Groups

| Group             | Parameters                            | Controls                                   |
|:------------------|:--------------------------------------|:-------------------------------------------|
| Na⁺ activation    | th_Na, p_Na, rate_Na, g_Na_max, E_Na | Spike threshold, upstroke speed, amplitude |
| Na⁺ inactivation  | rate_h1, rate_h2                      | Refractory period, firing bandwidth        |
| K⁺ repolarisation | th_K, p_K, rate_K, g_K_max, E_K      | Spike width, AHP depth, inter-spike trajectory |
| Leak & adaptation | g_L, v_rest, g_adapt, rate_slow, C_m | Resting potential, slow adaptation         |

---

## Citation

If you use this model in your research, please cite:

```
Silvera, P. G. (2026). Two-Exponential Neuron Model with Emergent Sigmoid Inactivation
and Rich Firing Phenotypes [Model]. Zenodo.
https://doi.org/10.5281/zenodo.19011537
```
