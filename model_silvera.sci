// ============================================================================
// Silvera Neuron Model — Scilab Implementation
// ============================================================================
// Copyright (C) 2026 Paolo Giovanni Silvera
// ORCID   : 0009-0005-9234-1818
// DOI     : 10.5281/zenodo.19011537
//
// License : Creative Commons Attribution-NonCommercial-ShareAlike 4.0
//           International (CC BY-NC-SA 4.0)
//           https://creativecommons.org/licenses/by-nc-sa/4.0/
//           Free for academic and research use.
//           Commercial use requires explicit written permission from the author.
//
// DISCLAIMER:
// This software is provided "as is", without warranty of any kind,
// express or implied, including but not limited to the warranties of
// merchantability, fitness for a particular purpose, and non-infringement.
// In no event shall the author be liable for any claim, damages, or other
// liability arising from, out of, or in connection with the software or
// the use thereof. This model is intended for research and educational
// purposes only. Results should be validated against experimental data
// before any clinical or commercial application.
//
// Description : Single-compartment continuous dynamical system (RS phenotpe).
//               5 ODEs, 17 parameters, 2 exponential evaluations per step.
//               No discontinuous reset — continuous limit cycle dynamics.
// ============================================================================

// ==========================================
// 1. PARAMETERS
// ==========================================

// Simulation
dt      = 0.05   // temporal timestep (ms)
tmax    = 600.0  // simulation duration (ms)
n_steps = int(ceil(tmax / dt))

// Stimulus
stim_t0  = 20.0   // start (ms)
stim_t1  = 560.0  // end (ms)
stim_amp = 1.32   // injected current amplitude (pA)

// Membrane
C_m    = 1.0     // membrane capacitance (µF/cm²)
g_L    = 0.04    // leak conductance (mS/cm²)
v_rest = -67.04  // resting potential (mV)

// Slow adaptation
rate_slow = 0.003  // adaptation time constant (ms⁻¹)
g_adapt   = 1.5    // adaptation channel gain

// Sodium channel (Na)
E_Na     = 65.0    // Na reversal potential (mV)
th_Na    = -51.44  // activation threshold (mV)
g_Na_max = 15.84   // max Na conductance (mS/cm²)
rate_Na  = 2.5     // m activation rate (ms⁻¹)
p_Na     = 1.0     // activation sigmoid slope

// Sodium inactivation (h)
rate_h1 = 13.884  // inactivation rate (depends on m)
rate_h2 = 3.975   // recovery rate

// Potassium channel (K)
E_K     = -91.483  // K reversal potential (mV)
th_K    = -40.0    // activation threshold (mV)
p_K     = 2.0      // activation sigmoid slope
g_K_max = 2.329    // max K conductance (mS/cm²)
rate_K  = 1.0      // n activation rate (ms⁻¹)


// ==========================================
// 2. SIGMOID FUNCTION
// ==========================================
// sigmoid(v, th, p) = 1 / (1 + exp(-(v - th) * p))
// v: current voltage, th: threshold, p: slope

function y = sigmoid(v, th, p)
    arg = -(v - th) * p;
    if arg >  100 then arg =  100; end  // overflow protection
    if arg < -100 then arg = -100; end
    y = 1.0 / (1.0 + exp(arg));
endfunction

// ==========================================
// 3. DIFFERENTIAL EQUATIONS
// ==========================================
// State vector y = [V, m, n, h, u]:
//   V: membrane potential (mV)
//   m: Na activation   — controls Na channel opening (0-1)
//   n: K  activation   — controls K  channel opening (0-1)
//   h: Na inactivation — reduces Na current during spike (0-1)
//   u: slow adaptation current (pA)

function dy = get_rates(t, y, I_stim)
    v = y(1); m = y(2); n = y(3); h = y(4); u = y(5);

    // Ionic currents
    I_Na   = g_Na_max * m * (1 - h) * (v - E_Na);   // sodium (inactivated by h)
    I_K    = g_K_max  * n           * (v - E_K);    // potassium current
    I_leak = g_L                    * (v - v_rest); // leak current

    // Steady-state gate values
    m_inf = sigmoid(v, th_Na, p_Na)  // Na activation at equilibrium
    n_inf = sigmoid(v, th_K,  p_K)   // K  activation at equilibrium

    dy = zeros(5, 1);
    dy(1) = (-I_leak - I_Na - I_K - u + I_stim) / C_m; // dV/dt
    dy(2) = (m_inf - m) * rate_Na;                     // dm/dt: Na activation
    dy(3) = (n_inf - n) * rate_K;                      // dn/dt: K activation
    dy(4) = (1 - h) * (m * rate_h1 - h * rate_h2);     // dh/dt: Na inactivation
    dy(5) = (g_adapt * (v - v_rest) - u) * rate_slow;  // du/dt: adaptation
endfunction

// ==========================================
// 4. SIMULATION (RK4)
// ==========================================

// Output arrays allocation
t_trace = zeros(n_steps, 1);
v_trace = zeros(n_steps, 1);
m_trace = zeros(n_steps, 1);
n_trace = zeros(n_steps, 1);
h_trace = zeros(n_steps, 1);
u_trace = zeros(n_steps, 1);

// Initial conditions: neuron at rest, all gates closed
y         = [v_rest; 0.0; 0.0; 0.0; 0.0];
current_t = 0;

printf("Starting RK4 simulation...\n");

for i = 1:n_steps
    // Injected current during stimulus window
    I_inj = stim_amp*(current_t>=stim_t0 & current_t<stim_t1);

    // Save traces at current step
    t_trace(i) = current_t;
    v_trace(i) = y(1);
    m_trace(i) = y(2);
    n_trace(i) = y(3);
    h_trace(i) = y(4);
    u_trace(i) = y(5);

    // Fixed-step RK4 integration
    k1 = get_rates(current_t,        y,           I_inj);
    k2 = get_rates(current_t + dt/2, y + k1*dt/2, I_inj);
    k3 = get_rates(current_t + dt/2, y + k2*dt/2, I_inj);
    k4 = get_rates(current_t + dt,   y + k3*dt,   I_inj);

    y         = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
    current_t = current_t + dt;
end

printf("Simulation complete.\n");

// ==========================================
// 5. PLOTTING
// ==========================================
drawlater();

// Membrane potential
subplot(3, 1, 1);
plot(t_trace, v_trace, 'k', 'LineWidth', 1);
xgrid(1);
xtitle("Membrane potential", "Tempo (ms)", "V (mV)");
gca().data_bounds = [0, min(v_trace)-5; tmax, max(v_trace)+10];

// Gating variables
subplot(3, 1, 2);
plot(t_trace, m_trace, 'b');             // Na Activation (m)
plot(t_trace, h_trace, 'r');             // Na Inactivation (h)
plot(t_trace, n_trace, 'g');             // K Activation (n)
legend(['m (Na activation)'; 'h (Na inactivation)'; 'n (K activation)']);
xgrid(1);
xtitle("Gating variables", "Time (ms)", "");

// Adaptation current
subplot(3, 1, 3);
plot(t_trace, u_trace, 'b', 'LineWidth', 1);
xgrid(1);
xtitle("Adaptation current", "Time (ms)", "u (pA)");

drawnow();
