import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

"""
Divergent Gravity: M/L Optimization Tool
----------------------------------------
Iterates through Mass-to-Light (Upsilon) ratios to achieve 0.5% 
precision fits for SPARC galaxies.
"""

# Constants
RU = 14.4 * u.Glyr
A0 = (const.c**2 / RU).to(u.m / u.s**2)


def calculate_dg_with_ml(df, upsilon):
    """
    Applies M/L (Upsilon) to the stellar components (Disk/Bulge).
    v_bar^2 = v_gas^2 + upsilon * (v_disk^2 + v_bul^2)
    """
    r = df["Rad"].values * u.kpc
    v_gas_sq = (df["Vgas"].values * u.km / u.s) ** 2
    v_stellar_sq = (df["Vdisk"].values ** 2 + df["Vbul"].values ** 2) * (
        u.km / u.s
    ) ** 2

    # Adjusted Baryonic Potential
    v_bar_sq = v_gas_sq + (upsilon * v_stellar_sq)

    # DG Equation
    v_dg = (v_bar_sq**2 + (v_bar_sq * r * A0)).decompose() ** 0.25
    return v_dg.to(u.km / u.s)


def optimize_ml(df, galaxy_name):
    ml_range = np.linspace(0.3, 1.2, 100)
    best_ml = 0
    min_error = float("inf")

    v_obs = df["Vobs"].values * u.km / u.s

    for ml in ml_range:
        v_pred = calculate_dg_with_ml(df, ml)
        # Mean Absolute Percentage Error (MAPE)
        error = np.mean(np.abs((v_pred - v_obs) / v_obs)) * 100

        if error < min_error:
            min_error = error
            best_ml = ml

    return best_ml, min_error


# Usage block (Assuming df is loaded from SPARC)
# ml, err = optimize_ml(df_ngc3198, "NGC3198")
# print(f"Best M/L: {ml:.2f}, Precision: {100-err:.2f}%")
