# =============================================================================
# Silvera Neuron Model — Python Implementation
# =============================================================================
# Copyright (C) 2026 Paolo Giovanni Silvera
# ORCID   : 0009-0005-9234-1818
# DOI     : 10.5281/zenodo.19011537
#
# License : Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#           International (CC BY-NC-SA 4.0)
#           https://creativecommons.org/licenses/by-nc-sa/4.0/
#           Free for academic and research use.
#           Commercial use requires explicit written permission from the author.
#
# DISCLAIMER:
# This software is provided "as is", without warranty of any kind,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
# In no event shall the author be liable for any claim, damages, or other
# liability arising from, out of, or in connection with the software or
# the use thereof. This model is intended for research and educational
# purposes only. Results should be validated against experimental data
# before any clinical or commercial application.
#
# Description : Single-compartment continuous dynamical system (RS phenotype).
#               5 ODEs, 17 parameters, 2 exponential evaluations per step.
#               No discontinuous reset — continuous limit cycle dynamics.
# Dependencies: numpy >= 1.21, matplotlib >= 3.4
# Usage       : python model_silvera.py
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS
# ==========================================

# Simulation
dt      = 0.05   # temporal timestep (ms)
tmax    = 600.0  # simulation duration (ms)
n_steps = int(np.ceil(tmax / dt))

# Stimulus
stim_t0  = 20.0   # start (ms)
stim_t1  = 560.0  # end (ms)
stim_amp = 1.32   # injected current amplitude (pA)

# Membrane
C_m    = 1.0     # membrane capacitance (µF/cm²)
g_L    = 0.04    # leak conductance (mS/cm²)
v_rest = -67.04  # resting potential (mV)

# Slow adaptation
rate_slow = 0.003  # adaptation time constant (ms⁻¹)
g_adapt   = 1.5    # adaptation channel gain

# Sodium channel (Na)
E_Na     = 65.0    # Na reversal potential (mV)
th_Na    = -51.44  # activation threshold (mV)
g_Na_max = 15.84   # max Na conductance (mS/cm²)
rate_Na  = 2.5     # m activation rate (ms⁻¹)
p_Na     = 1.0     # activation sigmoid slope

# Sodium inactivation (h)
rate_h1 = 13.884  # inactivation rate (depends on m)
rate_h2 = 3.975   # recovery rate

# Potassium channel (K)
E_K     = -91.483  # K reversal potential (mV)
th_K    = -40.0    # activation threshold (mV)
p_K     = 2.0      # activation sigmoid slope
g_K_max = 2.329    # max K conductance (mS/cm²)
rate_K  = 1.0      # n activation rate (ms⁻¹)


# ==========================================
# 2. SIGMOID FUNCTION
# ==========================================
# sigmoid(v, th, p) = 1 / (1 + exp(-(v - th) * p))
# v: current voltage, th: threshold, p: slope

def sigmoid(v, th, p):
    arg = np.clip(-(v - th) * p, -100, 100)  # overflow protection
    return 1.0 / (1.0 + np.exp(arg))


# ==========================================
# 3. DIFFERENTIAL EQUATIONS
# ==========================================
# State vector y = [V, m, n, h, u]:
#   V: membrane potential (mV)
#   m: Na activation   — controls Na channel opening (0-1)
#   n: K  activation   — controls K  channel opening (0-1)
#   h: Na inactivation — reduces Na current during spike (0-1)
#   u: slow adaptation current (pA)

def get_rates(y, I_stim):
    v, m, n, h, u = y

    # Ionic currents
    I_Na   = g_Na_max * m * (1 - h) * (v - E_Na)  # sodium current (inactivated by h)
    I_K    = g_K_max  * n           * (v - E_K)    # potassium current
    I_leak = g_L                    * (v - v_rest)  # leak current

    # Steady-state gate values
    m_inf = sigmoid(v, th_Na, p_Na)  # Na activation at equilibrium
    n_inf = sigmoid(v, th_K,  p_K)   # K  activation at equilibrium

    dy = np.zeros(5)
    dy[0] = (-I_leak - I_Na - I_K - u + I_stim) / C_m  # dV/dt
    dy[1] = (m_inf - m) * rate_Na                      # dm/dt: Na activation
    dy[2] = (n_inf - n) * rate_K                       # dn/dt: K  activation
    dy[3] = (1 - h) * (m * rate_h1 - h * rate_h2)      # dh/dt: Na inactivation
    dy[4] = (g_adapt * (v - v_rest) - u) * rate_slow   # du/dt: adaptation

    return dy


# ==========================================
# 4. SIMULATION (RK4)
# ==========================================

# Output arrays
t_trace = np.zeros(n_steps)
v_trace = np.zeros(n_steps)
m_trace = np.zeros(n_steps)
n_trace = np.zeros(n_steps)
h_trace = np.zeros(n_steps)
u_trace = np.zeros(n_steps)

# Initial conditions: neuron at rest, all gates closed
y         = np.array([v_rest, 0.0, 0.0, 0.0, 0.0])
current_t = 0.0

print("Starting RK4 simulation...")

for i in range(n_steps):
    # Injected current during stimulus window
    I_inj = stim_amp if stim_t0 <= current_t <= stim_t1 else 0.0

    # Save traces at current step
    t_trace[i] = current_t
    v_trace[i] = y[0]
    m_trace[i] = y[1]
    n_trace[i] = y[2]
    h_trace[i] = y[3]
    u_trace[i] = y[4]

    # Fixed-step RK4 integration
    k1 = get_rates(y,           I_inj)
    k2 = get_rates(y + k1*dt/2, I_inj)
    k3 = get_rates(y + k2*dt/2, I_inj)
    k4 = get_rates(y + k3*dt,   I_inj)

    y         = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    current_t = current_t + dt

print("Simulation complete.")


# ==========================================
# 5. PLOTTING
# ==========================================

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Silvera Neuron Model — Regular Spiking (RS)', fontsize=12, fontweight='bold')

# Membrane potential
axes[0].plot(t_trace, v_trace, 'k', lw=1.0)
axes[0].set_ylabel('V (mV)')
axes[0].set_ylim(v_trace.min() - 5, v_trace.max() + 10)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Membrane potential', fontsize=10)

# Gating variables
axes[1].plot(t_trace, m_trace,           'b', lw=1.0, label='m (Na\u207a activation)')
axes[1].plot(t_trace, h_trace,           'r', lw=1.0, label='h (Na\u207a inactivation)')
axes[1].plot(t_trace, n_trace,           'g', lw=1.0, label='n (K\u207a activation)')
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(fontsize=8, loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Gating variables', fontsize=10)

# Adaptation current
axes[2].plot(t_trace, u_trace, 'b', lw=1.0)
axes[2].set_ylabel('u (pA)')
axes[2].set_xlabel('Time (ms)')
axes[2].grid(True, alpha=0.3)
axes[2].set_title('Adaptation current', fontsize=10)

plt.tight_layout()
plt.show()
