"""
plot_phenotypes.py
==================
Silvera Neuron Model — Phenotype Voltage Traces
================================================

Generates a publication-quality PDF figure showing voltage traces and
aligned spike overlays for all phenotypes defined in the accompanying
XML configuration files.

Each panel displays:
  - Left  : full voltage trace V(t) with stimulus bar
  - Right : first 5 spikes superimposed and centred at their peak (±2.5 ms)

Simulation engine
-----------------
Conductance-based Hodgkin-Huxley-like dynamics with five state variables
(V, m, n, h, u), integrated with a 4th-order Runge-Kutta scheme.
Two exponential evaluations per time step (Na and K activation sigmoids).

Usage
-----
Place this script in the same directory as the XML parameter files, then run::

    python plot_phenotypes.py

Output: ``phenotype_traces.pdf``

Author : Paolo Giovanni Silvera
License: CC BY-NC-SA 4.0
DOI    : 10.5281/zenodo.19133651
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _sigmoid(v: float, threshold: float, slope: float) -> float:
    """Boltzmann-style sigmoid activation function.

    Parameters
    ----------
    v         : membrane potential (mV)
    threshold : half-activation voltage (mV)
    slope     : slope factor (mV⁻¹); higher values produce sharper transitions
    """
    arg = -(v - threshold) * slope
    arg = max(-100.0, min(100.0, arg))   # prevent overflow
    return 1.0 / (1.0 + np.exp(arg))


def _derivatives(
    v: float, m: float, n: float, h: float, u: float,
    I_stim: float,
    C_m: float, g_L: float, rate_slow: float, g_adapt: float,
    E_Na: float, th_Na: float, p_Na: float, g_Na_max: float, rate_Na: float,
    E_K: float,  th_K: float,  p_K: float,  g_K_max: float,  rate_K: float,
    rate_h1: float, rate_h2: float, v_rest: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Compute all state-variable derivatives for one time step.

    Returns
    -------
    dv, dm, dn, dh, du, I_Na, I_K
    """
    I_Na   = g_Na_max * m * (1.0 - h) * (v - E_Na)
    I_K    = g_K_max  * n             * (v - E_K)
    I_leak = g_L * (v - v_rest)

    dv = (-I_leak - I_Na - I_K - u + I_stim) / C_m

    m_inf = _sigmoid(v, th_Na, p_Na)
    n_inf = _sigmoid(v, th_K,  p_K)

    dm = (m_inf - m) * rate_Na
    dn = (n_inf - n) * rate_K
    dh = (1.0 - h) * (m * rate_h1 - h * rate_h2)
    du = (g_adapt * (v - v_rest) - u) * rate_slow

    return dv, dm, dn, dh, du, I_Na, I_K


def simulate_rk4(
    dt: float,
    tmax: float,
    I_array: np.ndarray,
    C_m: float, g_L: float, rate_slow: float, g_adapt: float,
    E_Na: float, th_Na: float, p_Na: float, g_Na_max: float, rate_Na: float,
    E_K: float,  th_K: float,  p_K: float,  g_K_max: float,  rate_K: float,
    rate_h1: float, rate_h2: float,
    noise_gain: float,
    v_rest: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Silvera model using a 4th-order Runge-Kutta scheme.

    Parameters
    ----------
    dt        : integration time step (ms)
    tmax      : total simulation duration (ms)
    I_array   : pre-computed stimulus current array (µA/cm²), length == n_steps
    noise_gain: standard deviation of additive Gaussian current noise

    Returns
    -------
    t_trace : time vector (ms)
    v_trace : membrane potential trace (mV)
    """
    n_steps = len(I_array)
    v_trace = np.zeros(n_steps)
    t_trace = np.linspace(0.0, tmax, n_steps)

    v, m, n, h, u = v_rest, 0.0, 0.0, 0.0, 0.0

    for i in range(n_steps):
        I = I_array[i]
        if noise_gain > 0.0:
            I += np.random.randn() * noise_gain

        args = (
            I,
            C_m, g_L, rate_slow, g_adapt,
            E_Na, th_Na, p_Na, g_Na_max, rate_Na,
            E_K,  th_K,  p_K,  g_K_max,  rate_K,
            rate_h1, rate_h2, v_rest,
        )

        k1 = _derivatives(v,                m,                n,                h,                u,                *args)
        k2 = _derivatives(v + dt/2*k1[0],   m + dt/2*k1[1],   n + dt/2*k1[2],   h + dt/2*k1[3],   u + dt/2*k1[4],   *args)
        k3 = _derivatives(v + dt/2*k2[0],   m + dt/2*k2[1],   n + dt/2*k2[2],   h + dt/2*k2[3],   u + dt/2*k2[4],   *args)
        k4 = _derivatives(v + dt  *k3[0],   m + dt  *k3[1],   n + dt  *k3[2],   h + dt  *k3[3],   u + dt  *k3[4],   *args)

        v += (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        m += (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        n += (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        h += (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        u += (dt / 6.0) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
        h = max(0.0, min(1.0, h))

        v_trace[i] = v

    return t_trace, v_trace


# ---------------------------------------------------------------------------
# XML parameter loading
# ---------------------------------------------------------------------------

def load_xml(path: str) -> tuple[dict, list[dict]]:
    """Parse a Silvera XML configuration file.

    Returns
    -------
    params : dict mapping parameter names to float values
    pulses : list of stimulus pulse dicts (start, dur, amp_start, amp_end)
    """
    root = ET.parse(path).getroot()
    params = {p.get('key'): float(p.get('value'))
              for p in root.findall('.//Param')}
    pulses = [
        {
            'start':     float(p.get('start')),
            'dur':       float(p.get('dur')),
            'amp_start': float(p.get('amp_start')),
            'amp_end':   float(p.get('amp_end')),
        }
        for p in root.findall('.//Pulse')
    ]
    return params, pulses


def build_current_array(params: dict, pulses: list[dict]) -> np.ndarray:
    """Convert pulse specifications into a sample-by-sample current array."""
    dt      = params['dt']
    tmax    = params['tmax']
    n_steps = int(np.ceil(tmax / dt))
    I_arr   = np.zeros(n_steps)
    for p in pulses:
        i0 = int(p['start'] / dt)
        i1 = min(i0 + int(p['dur'] / dt), n_steps)
        if i0 < n_steps:
            I_arr[i0:i1] = np.linspace(p['amp_start'], p['amp_end'], i1 - i0)
    return I_arr


def run(params: dict, pulses: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load parameters, build stimulus, and run RK4 simulation."""
    I_arr = build_current_array(params, pulses)
    t, v  = simulate_rk4(
        params['dt'], params['tmax'], I_arr,
        params['C_m'],  params['g_L'],    params['rate_slow'], params['g_adapt'],
        params['E_Na'], params['th_Na'],  params['p_Na'],      params['g_Na_max'], params['rate_Na'],
        params['E_K'],  params['th_K'],   params['p_K'],       params['g_K_max'],  params['rate_K'],
        params['rate_h1'], params['rate_h2'],
        params['noise_gain'], params['v_rest'],
    )
    return t, v, I_arr


# ---------------------------------------------------------------------------
# Spike detection and waveform extraction
# ---------------------------------------------------------------------------

def extract_spike_waveforms(
    t: np.ndarray,
    v: np.ndarray,
    dt: float,
    half_window_ms: float = 2.5,
    max_spikes: int = 5,
    threshold_mv: float = -20.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract aligned spike waveforms centred at their peak.

    Parameters
    ----------
    t               : time vector (ms)
    v               : voltage trace (mV)
    dt              : time step (ms)
    half_window_ms  : half-width of the extraction window (ms)
    max_spikes      : maximum number of spikes to extract
    threshold_mv    : minimum peak amplitude for spike detection (mV)

    Returns
    -------
    t_relative : time axis relative to spike peak (ms)
    waveforms  : list of voltage snippets, each of length 2*half_window + 1
    """
    half  = int(half_window_ms / dt)
    min_spacing = half  # minimum samples between consecutive peaks

    peaks: list[int] = []
    i = 1
    while i < len(v) - 1 and len(peaks) < max_spikes:
        if v[i] > threshold_mv and v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            lo = max(0, i - 3)
            hi = min(len(v) - 1, i + 3)
            peak_i = lo + int(np.argmax(v[lo:hi + 1]))
            if not peaks or (peak_i - peaks[-1]) > min_spacing:
                peaks.append(peak_i)
            i = peak_i + min_spacing
        else:
            i += 1

    t_relative = np.linspace(-half_window_ms, half_window_ms, 2 * half + 1)
    waveforms: list[np.ndarray] = []
    for pk in peaks:
        lo, hi = pk - half, pk + half + 1
        if lo >= 0 and hi <= len(v):
            waveforms.append(v[lo:hi])

    return t_relative, waveforms


# ---------------------------------------------------------------------------
# Phenotype configuration
# ---------------------------------------------------------------------------

# Human-readable labels for known XML stems
LABEL_MAP: dict[str, str] = {
    'RS6':           'RS – Regular Spiking',
    'FS6':           'FS – Fast Spiking',
    'CH6':           'CH – Chattering',
    'IB6':           'IB – Intrinsic Bursting',
    'LTS6':          'LTS – Low-Threshold Spiking',
    'REBOUND_SPIKE6':'Rebound Spike',
    'REBOUND_BURST6':'Rebound Burst',
    'FREQ_ADAPT':    'Freq. Adaptation',
    'RZ':            'Resonator',
    'NT':            'NT – Integrator',
}

# Colour palette for distinct phenotypes
PALETTE: list[str] = [
    '#e6000a', '#0057e3', '#00a800', '#ff7700',
    '#8800cc', '#00b8b8', '#cc0077', '#8b4000',
    '#006600', '#0000cc', '#cc6600', '#004488',
]

def discover_neurons(base_dir: str) -> list[tuple[str, str, str]]:
    """Discover XML files in *base_dir* and return (path, label, colour) tuples."""
    xml_files = sorted(glob.glob(os.path.join(base_dir, '*.xml')))
    neurons = []
    for i, path in enumerate(xml_files):
        stem  = os.path.splitext(os.path.basename(path))[0]
        label = LABEL_MAP.get(stem, stem.replace('_', ' '))
        color = PALETTE[i % len(PALETTE)]
        neurons.append((path, label, color))
    return neurons


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_phenotypes(
    neurons: list[tuple[str, str, str]],
    output_path: str = 'phenotype_traces.pdf',
    half_window_ms: float = 2.5,
    n_spikes: int = 5,
) -> None:
    """Simulate all phenotypes and render the publication figure.

    Layout
    ------
    Two phenotypes per row.  Each phenotype occupies two columns:
      [3-wide main trace] [1-wide spike overlay]

    Parameters
    ----------
    neurons       : list of (xml_path, label, colour) from ``discover_neurons``
    output_path   : destination PDF file
    half_window_ms: half-width of spike overlay window (ms)
    n_spikes      : number of spikes to overlay per panel
    """
    n_per_row = 2
    n_neurons = len(neurons)
    n_rows    = (n_neurons + n_per_row - 1) // n_per_row

    fig = plt.figure(figsize=(18, 3.8 * n_rows))
    fig.suptitle(
        'Voltage traces of different firing phenotypes – Silvera neuron model',
        fontsize=14, fontweight='bold',
    )

    gs = gridspec.GridSpec(
        n_rows, n_per_row * 2, figure=fig,
        width_ratios=[3, 1] * n_per_row,
        hspace=0.55, wspace=0.35,
    )

    for idx, (path, label, color) in enumerate(neurons):
        params, pulses = load_xml(path)
        print(
            f'[{idx + 1:02d}/{n_neurons}]  Simulating: {label:<35}'
            f'tmax = {params["tmax"]:.0f} ms',
            flush=True,
        )
        t, v, _ = run(params, pulses)

        row      = idx // n_per_row
        col_main = (idx % n_per_row) * 2
        col_spk  = col_main + 1

        # ── Main voltage trace ────────────────────────────────────────────
        ax_main = fig.add_subplot(gs[row, col_main])
        ax_main.plot(t, v, color=color, linewidth=0.9)
        ax_main.set_title(label, fontsize=9, fontweight='bold')
        ax_main.set_xlabel('Time (ms)', fontsize=7)
        ax_main.set_ylabel('V (mV)',    fontsize=7)
        ax_main.tick_params(labelsize=6)
        ax_main.set_xlim(0.0, params['tmax'])
        ax_main.autoscale(axis='y')

        # Stimulus bar at the bottom of the panel
        ymin, ymax = ax_main.get_ylim()
        bar_height = (ymax - ymin) * 0.03
        for p in pulses:
            ax_main.axhspan(
                ymin, ymin + bar_height,
                xmin=p['start'] / params['tmax'],
                xmax=min((p['start'] + p['dur']) / params['tmax'], 1.0),
                color='gray', alpha=0.45, linewidth=0,
            )
        ax_main.autoscale(axis='y')

        # ── Spike overlay panel ───────────────────────────────────────────
        ax_spk = fig.add_subplot(gs[row, col_spk])
        t_rel, waveforms = extract_spike_waveforms(
            t, v, params['dt'],
            half_window_ms=half_window_ms,
            max_spikes=n_spikes,
        )

        if waveforms:
            alphas = np.linspace(0.3, 1.0, len(waveforms))
            for k, wf in enumerate(waveforms):
                ax_spk.plot(t_rel, wf, color=color,
                            linewidth=0.8, alpha=float(alphas[k]))
            ax_spk.axvline(0.0, color='k', linewidth=0.5,
                           linestyle='--', alpha=0.4)
        else:
            ax_spk.text(0.5, 0.5, 'no spikes detected',
                        ha='center', va='center',
                        transform=ax_spk.transAxes,
                        fontsize=7, color='gray')

        ax_spk.set_xlabel('t \u2013 t\u2080 (ms)', fontsize=7)
        ax_spk.set_ylabel('V (mV)',                 fontsize=7)
        ax_spk.tick_params(labelsize=6)
        ax_spk.set_xlim(-half_window_ms, half_window_ms)
        ax_spk.set_title(f'first {n_spikes} spikes', fontsize=7, style='italic')

    print(f'\nRendering complete. Saving to: {output_path}')
    plt.savefig(output_path, bbox_inches='tight')
    print('Done.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    neurons  = discover_neurons(base_dir)

    if not neurons:
        raise FileNotFoundError(
            f'No XML configuration files found in: {base_dir}'
        )

    print(f'Found {len(neurons)} phenotype(s):\n')
    for _, label, _ in neurons:
        print(f'  • {label}')
    print()

    plot_phenotypes(
        neurons,
        output_path=os.path.join(base_dir, 'phenotype_traces.pdf'),
    )
