# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import sph_harm, genlaguerre
import plotly.express as px

# ----------------------------
# Helper Functions
# ----------------------------

def radial_wavefunction(n, l, r, a0=1.0):
    """Compute R_{n,l}(r) for hydrogen atom (in atomic units)."""
    rho = 2 * r / (n * a0)
    L = genlaguerre(n - l - 1, 2 * l + 1)
    norm = np.sqrt(
        (2.0 / (n * a0))**3 *
        np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l))
    )
    R = norm * np.exp(-rho / 2) * rho**l * L(rho)
    return R

def hydrogen_psi(n, l, m, r, theta, phi, a0=1.0):
    """Compute full wavefunction ψ_{n,l,m}(r,θ,φ)."""
    R = radial_wavefunction(n, l, r, a0)
    Y = sph_harm(m, l, phi, theta)
    return R * Y

def probability_density(n, l, m, r, theta, phi):
    """Compute |ψ|^2."""
    psi = hydrogen_psi(n, l, m, r, theta, phi)
    return np.abs(psi)**2

# ----------------------------
# Load Precomputed Dataset
# ----------------------------
cluster_map = {}
data_records = []

n_max = 4
for n in range(1, n_max + 1):
    for l in range(0, n):
        for m in range(-l, l + 1):
            # Analytical <r> (in a0)
            r_exp = 0.5 * (3 * n**2 - l * (l + 1))
            r2_exp = 0.5 * n**2 * (5 * n**2 + 1 - 3 * l * (l + 1))
            sigma_r = np.sqrt(r2_exp - r_exp**2)
            # Assign cluster based on your results
            if n <= 2 or (n == 3 and l <= 1):
                cluster = 0
            elif n == 4 and l == 3:
                cluster = 2
            else:
                cluster = 1
            cluster_map[(n, l, m)] = cluster
            data_records.append({
                "n": n, "l": l, "m": m,
                "<r> (a0)": round(r_exp, 2),
                "σ_r (a0)": round(sigma_r, 2),
                "Cluster": cluster,
                "Cluster Name": ["Compact", "Intermediate", "Diffuse"][cluster]
            })

df_features = pd.DataFrame(data_records)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Hydrogen Orbital Explorer", layout="wide")
st.title("🔍 Hydrogen Atomic Orbital Explorer")
st.markdown("""
Explore the quantum structure of the hydrogen atom through interactive 3D visualizations and data-driven insights.  
This tool is based on computational physics and machine learning analysis of orbital geometry.
""")

# Sidebar for quantum number selection
st.sidebar.header("Orbital Selection")
n = st.sidebar.selectbox("Principal quantum number (n)", options=list(range(1, 5)), index=0)
l_options = list(range(0, n))
l = st.sidebar.selectbox("Azimuthal quantum number (l)", options=l_options, index=0)
m_options = list(range(-l, l + 1))
m = st.sidebar.selectbox("Magnetic quantum number (m)", options=m_options, index=l)

# Retrieve precomputed features
orbital_key = (n, l, m)
row = df_features[(df_features['n'] == n) & (df_features['l'] == l) & (df_features['m'] == m)].iloc[0]

# Display features
st.subheader(f"Orbital: ψ_{{{n},{l},{m}}}")
col1, col2, col3 = st.columns(3)
col1.metric("⟨r⟩ (a₀)", row["<r> (a0)"])
col2.metric("σᵣ (a₀)", row["σ_r (a0)"])
col3.metric("Cluster", row["Cluster Name"])

# 3D Visualization
st.subheader("3D Probability Density |ψ|²")
phi = np.linspace(0, 2 * np.pi, 30)
theta = np.linspace(0, np.pi, 30)
phi, theta = np.meshgrid(phi, theta)

# Use a fixed radial slice for visualization (e.g., at r = ⟨r⟩)
r_fixed = row["<r> (a0)"]
psi_sq = probability_density(n, l, m, r_fixed, theta, phi)
x = r_fixed * np.sin(theta) * np.cos(phi)
y = r_fixed * np.sin(theta) * np.sin(phi)
z = r_fixed * np.cos(theta)

fig = go.Figure(data=go.Surface(
    x=x, y=y, z=z, surfacecolor=psi_sq,
    colorscale='Viridis', colorbar=dict(title="|ψ|²")
))
fig.update_layout(
    scene=dict(
        xaxis_title='x (a₀)',
        yaxis_title='y (a₀)',
        zaxis_title='z (a₀)',
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)
st.plotly_chart(fig, use_container_width=True)

# Show full dataset (optional)
with st.expander("📊 View Full Dataset (n=1 to 4)"):
    st.dataframe(df_features.style.format({
        "<r> (a0)": "{:.2f}",
        "σ_r (a0)": "{:.2f}"
    }))

st.markdown("""
---
**Note**: This application is part of a research project on data-driven characterization of hydrogen orbitals.  
All simulations are in atomic units (a₀ = 1). Code and data are open-source.
""")
