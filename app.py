# app.py (versi tanpa scipy)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# Load Precomputed Dataset
# ----------------------------
df = pd.read_csv("orbital_features.csv", sep=';', header=0)

# Map (n,l,m) to cluster name
cluster_names = {0: "Compact", 1: "Intermediate", 2: "Diffuse"}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Hydrogen Orbital Explorer", layout="wide")
st.title("🔍 Hydrogen Atomic Orbital Explorer")
st.markdown("""
Interactive visualization of hydrogen atomic orbitals based on precomputed quantum features.  
All data derived from numerical solutions of the Schrödinger equation.
""")

# Sidebar selection
n = st.sidebar.selectbox("Principal quantum number (n)", options=sorted(df["n"].unique()), index=0)
l_options = sorted(df[df["n"] == n]["l"].unique())
l = st.sidebar.selectbox("Azimuthal quantum number (l)", options=l_options, index=0)
m_options = sorted(df[(df["n"] == n) & (df["l"] == l)]["m"].unique())
m = st.sidebar.selectbox("Magnetic quantum number (m)", options=m_options, index=len(m_options)//2)

# Get data
row = df[(df["n"] == n) & (df["l"] == l) & (df["m"] == m)].iloc[0]

st.subheader(f"Orbital: ψ_{{{int(n)},{int(l)},{int(m)}}}")
col1, col2, col3 = st.columns(3)
col1.metric("⟨r⟩ (a₀)", f"{row['r_exp_analytical']:.2f}")
col2.metric("σᵣ (a₀)", f"{row['sigma_r']:.2f}")
col3.metric("Cluster", cluster_names.get(row["cluster"], "Unknown"))
cluster_name = cluster_names.get(row["cluster"], "Unknown")

# Simple 3D visualization (approximate shape only)
st.subheader("3D Orbital Shape (Approximation)")
phi = np.linspace(0, 2 * np.pi, 30)
theta = np.linspace(0, np.pi, 30)
phi, theta = np.meshgrid(phi, theta)

if l == 0:
    angular = np.ones_like(theta)
elif l == 1:
    if m == 0:
        angular = np.cos(theta)**2
    else:
        angular = np.sin(theta)**2
elif l == 2:
    if m == 0:
        angular = (3*np.cos(theta)**2 - 1)**2
    elif abs(m) == 1:
        angular = (np.sin(2*theta))**2
    else:  # |m| == 2
        angular = (np.sin(theta)**2)**2
else:
    # For l>=3, use generic lobe structure
    angular = np.abs(np.sin((l - abs(m)) * theta) * np.cos(m * phi)) + 0.1

# Radial scale based on <r>
r_scale = row["<r>_analitik"]
x = r_scale * angular * np.sin(theta) * np.cos(phi)
y = r_scale * angular * np.sin(theta) * np.sin(phi)
z = r_scale * angular * np.cos(theta)

fig = go.Figure(data=go.Surface(
    x=x, y=y, z=z, surfacecolor=angular,
    colorscale='Viridis', showscale=False
))
fig.update_layout(
    scene=dict(
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)
st.plotly_chart(fig, use_container_width=True)

# Show full dataset
with st.expander("📊 Full Dataset (n=1 to 4)"):
    st.dataframe(df.style.format({
        "<r>_analitik": "{:.3f}",
        "<r>_numerik": "{:.3f}",
        "sigma_r": "{:.3f}",
        "r_mode": "{:.3f}",
        "relative_error": "{:.1%}"
    }))

st.markdown("""
---
**Note**: This app uses precomputed quantum mechanical data.  
No real-time wavefunction calculation is performed.  
Based on research: "Data-Driven Characterization of Hydrogen Atomic Orbitals".
""")
