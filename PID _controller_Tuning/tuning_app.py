import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks, TransferFunction, bode
from scipy.optimize import minimize
from scipy.io import loadmat

st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 5])  # Adjust column width for positioning
with col1:
    st.image("logo.jpg", width=150)  # Replace "logo.jpg" with your actual logo path
    st.markdown("""
        <div style="font-size: 14px; font-weight: bold; margin-top: 5px;">
             Masoud Bakhshi<br>
             <a href="mailto:masoud.bakhshi@volvo.com">masoud.bakhshi@volvo.com</a>
        </div>
        """, unsafe_allow_html=True)
with col2:
    st.title("PI Controller Tuning and Stability App")

# Sidebar for user instructions
st.sidebar.header("Instructions for Optimal Tuning")
st.sidebar.write("""
1. **Choose Mode**: Select whether you are running the controller for the first time or tuning it based on applied gains.
2. **Provide Ultimate Gain Values**: If using Ziegler-Nichols, enter the Ultimate Gain values for Q and D axes before uploading the .mat file.
3. **Upload .mat File**: Upload a .mat file with specific variable names for reference and measured currents for both Q and D axes.
4. **Automatic Gain Calculation**: App calculates optimal Kp, Ki for both axes based on RMS error minimization.
5. **Stability Assessment**: The app will display stability results based on Bode and Nyquist analysis.
""")

# Controller settings input section in the sidebar
st.sidebar.header("Controller Settings")
T = st.sidebar.number_input("Sampling Time (seconds)", min_value=0.0001, value=0.000125, step=0.000001, format="%.6f")

# User selection: First time tuning or re-tuning based on imported data
tuning_mode = st.sidebar.radio("Select Tuning Mode", ("First Time Tuning (Ziegler-Nichols)", "Re-tuning based on Imported Data"))

# Sidebar inputs for Ziegler-Nichols initial tuning with 4 decimal places
if tuning_mode == "First Time Tuning (Ziegler-Nichols)":
    K_u_q = st.sidebar.number_input("Ultimate Gain (K_u) for Q-axis:", min_value=0.0, step=0.0001, format="%.4f")
    K_u_d = st.sidebar.number_input("Ultimate Gain (K_u) for D-axis:", min_value=0.0, step=0.0001, format="%.4f")

# Function to read .mat file and extract specified variables into a DataFrame
def read_mat_file(file):
    mat_data = loadmat(file)
    try:
        ref_q_current = np.squeeze(mat_data['dmtc_RefCurrEm_RefCurrEmQ']['signals'][0, 0]['values'][0, 0])
        measured_q_current = np.squeeze(mat_data['dmft_ActCurrEmQ']['signals'][0, 0]['values'][0, 0])
        ref_d_current = np.squeeze(mat_data['dmtc_RefCurrEm_RefCurrEmD']['signals'][0, 0]['values'][0, 0])
        measured_d_current = np.squeeze(mat_data['dmft_ActCurrEmD']['signals'][0, 0]['values'][0, 0])
    except KeyError as e:
        st.error(f"Error: The expected field '{e.args[0]}' was not found in the .mat file.")
        return None
    
    data = pd.DataFrame({
        'Reference Q Current': ref_q_current,
        'Measured Q Current': measured_q_current,
        'Reference D Current': ref_d_current,
        'Measured D Current': measured_d_current
    })
    return data

# Function to calculate Kp and Ki based on Ziegler-Nichols method
def calculate_zn_tuning(K_u, P_u):
    Kp = 0.45 * K_u
    Ki = Kp / (P_u / 2) if P_u > 0 else 0.0
    return Kp, Ki

# Function to calculate ultimate period (P_u) from oscillations
def calculate_ultimate_period(data, column_name):
    current_data = np.array(data[column_name]).flatten()
    peaks, _ = find_peaks(current_data)
    if len(peaks) > 1:
        periods = np.diff(peaks)
        P_u = np.mean(periods)
    else:
        P_u = 0
    return P_u

# Function to optimize Kp and Ki based on RMS error minimization
def optimize_pi_params(ref_current, measured_current):
    def objective(params):
        Kp, Ki = params
        error = ref_current - measured_current
        integral_error = np.cumsum(error) * T
        control_signal = Kp * error + Ki * integral_error
        rms_error = np.sqrt(np.mean((ref_current - control_signal) ** 2))
        return rms_error

    initial_guess = [1.0, 0.1]
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    
    result = minimize(objective, initial_guess, bounds=bounds)
    return result.x if result.success else (0, 0)

# Function to plot Nyquist plot and assess stability
def plot_nyquist_and_assess_stability(system):
    w, H = system.freqresp()
    real_part = H.real.flatten()
    imag_part = H.imag.flatten()
    
    fig_nyquist = go.Figure()
    fig_nyquist.add_trace(go.Scatter(x=real_part, y=imag_part, mode='lines', name='Nyquist Plot'))
    fig_nyquist.add_trace(go.Scatter(x=real_part, y=-imag_part, mode='lines', line=dict(dash='dash')))
    fig_nyquist.update_layout(
        title="Nyquist Plot",
        xaxis=dict(title="Real"),
        yaxis=dict(title="Imaginary"),
        height=300,
        width=600
    )
    st.plotly_chart(fig_nyquist, use_container_width=True)
    
    encirclements = np.sum((real_part[:-1] < -1) & (real_part[1:] > -1) & (imag_part[1:] * imag_part[:-1] < 0))
    stability = "Stable" if encirclements == 0 else "Unstable"
    st.markdown(f"<h3 style='color: {'green' if stability == 'Stable' else 'red'}; font-weight: bold;'>Stability Conclusion: {stability}</h3>", unsafe_allow_html=True)

# Function to plot Bode plot
def plot_bode_and_calculate_stability(system):
    w, mag, phase = bode(system)
    fig_bode = go.Figure()
    fig_bode.add_trace(go.Scatter(x=w, y=mag, mode='lines', name='Magnitude (dB)'))
    fig_bode.add_trace(go.Scatter(x=w, y=phase, mode='lines', name='Phase (degrees)', yaxis="y2"))
    fig_bode.update_layout(
        title="Bode Plot",
        xaxis=dict(title="Frequency (rad/s)", type="log"),
        yaxis=dict(title="Magnitude (dB)"),
        yaxis2=dict(title="Phase (degrees)", overlaying="y", side="right"),
        height=300,
        width=600
    )
    st.plotly_chart(fig_bode, use_container_width=True)

    gain_margin = None
    phase_margin = None
    for i, p in enumerate(phase):
        if p <= -180:
            gain_margin = -mag[i]
            break
    for i, m in enumerate(mag):
        if m >= 0:
            phase_margin = 180 + phase[i]
            break

    return phase_margin if phase_margin is not None else "N/A", gain_margin if gain_margin is not None else "N/A"

# Upload .mat file
uploaded_file = st.file_uploader("Upload .mat file", type=["mat"])

if uploaded_file:
    data = read_mat_file(uploaded_file)
    
    if data is not None:
        ref_q_current = data['Reference Q Current']
        measured_q_current = data['Measured Q Current']
        ref_d_current = data['Reference D Current']
        measured_d_current = data['Measured D Current']

        st.subheader("Current Responses")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=data.index, y=ref_q_current, mode='lines', name='Reference Q Current', line=dict(color='blue')))
        fig_q.add_trace(go.Scatter(x=data.index, y=measured_q_current, mode='lines', name='Measured Q Current', line=dict(color='orange')))
        fig_q.update_layout(title="Q-Axis Currents", xaxis_title="Sample Index", yaxis_title="Current (A)")
        st.plotly_chart(fig_q, use_container_width=True)

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=data.index, y=ref_d_current, mode='lines', name='Reference D Current', line=dict(color='green')))
        fig_d.add_trace(go.Scatter(x=data.index, y=measured_d_current, mode='lines', name='Measured D Current', line=dict(color='red')))
        fig_d.update_layout(title="D-Axis Currents", xaxis_title="Sample Index", yaxis_title="Current (A)")
        st.plotly_chart(fig_d, use_container_width=True)

        if tuning_mode == "First Time Tuning (Ziegler-Nichols)":
            P_u_q = calculate_ultimate_period(data, 'Measured Q Current')
            P_u_d = calculate_ultimate_period(data, 'Measured D Current')
            
            st.sidebar.write(f"Calculated Ultimate Period (P_u) for Q-axis: {P_u_q:.2f} samples")
            st.sidebar.write(f"Calculated Ultimate Period (P_u) for D-axis: {P_u_d:.2f} samples")

            Kp_q, Ki_q = calculate_zn_tuning(K_u_q, P_u_q)
            Kp_d, Ki_d = calculate_zn_tuning(K_u_d, P_u_d)
        elif tuning_mode == "Re-tuning based on Imported Data":
            Kp_q, Ki_q = optimize_pi_params(ref_q_current, measured_q_current)
            Kp_d, Ki_d = optimize_pi_params(ref_d_current, measured_d_current)

        system_q = TransferFunction([Kp_q, Ki_q], [1, Kp_q, Ki_q])

        st.subheader("Calculated PI Parameters")
        st.write(f"Q-Axis: Kp = {Kp_q:.2f}, Ki = {Ki_q:.2f}")
        st.write(f"D-Axis: Kp = {Kp_d:.2f}, Ki = {Ki_d:.2f}")

        st.header("Stability Analysis")
        phase_margin, gain_margin = plot_bode_and_calculate_stability(system_q)
        st.write(f"Phase Margin: {phase_margin} degrees, Gain Margin: {gain_margin} dB")
        
        plot_nyquist_and_assess_stability(system_q)
