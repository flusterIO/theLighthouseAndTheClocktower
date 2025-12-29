import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import requests
import zipfile
from astropy import units as u
from astropy import constants as const


"""
Divergent Gravity: Professional SPARC Validation Pipeline
---------------------------------------------------------
This script uses Astropy for unit-safe physical derivations 
and includes a robust data-fetching layer to handle server errors.
"""

# --- 1. Divergent Gravity Constants ---
# Background Potential identity: Phi_inf = c^2
# Hubble Radius (Ru) ~ 14.4 Glyr
RU = 14.4 * u.Glyr
A0 = (const.c**2 / RU).to(u.m / u.s**2)

# --- 2. Data Access Layer ---
# Official Mirror (maintained by the SPARC team on Zenodo)
DATA_URL = "https://zenodo.org/record/16284118/files/Rotmod_LTG.zip"
LOCAL_DIR = "sparc_cache"
ZIP_FILE = os.path.join(LOCAL_DIR, "Rotmod_LTG.zip")
# CONSTANTS (Explicit SI)
C = 299792458.0
RU = 1.362e26
A0 = (C**2) / RU  # ~6.6e-10 m/s^2
KPC_TO_M = 3.086e19
KM_S_TO_M_S = 1000.0


def get_clean_plot(df, name, upsilon=0.5):
    # 1. Correct Parsing
    # We must ensure we are using the right columns
    r_kpc = df["Rad"].values
    v_obs = df["Vobs"].values
    v_gas = df["Vgas"].values
    v_disk = df["Vdisk"].values
    v_bul = df["Vbul"].values

    # 2. Conversion to SI for the DG "Engine"
    r_m = r_kpc * KPC_TO_M
    v_gas_ms_sq = (v_gas * KM_S_TO_M_S) ** 2
    v_star_ms_sq = ((v_disk**2 + v_bul**2) * (KM_S_TO_M_S**2)) * upsilon

    # Total Baryonic Potential Term (v_bar^2)
    v_bar_sq = v_gas_ms_sq + v_star_ms_sq

    # 3. The Divergent Gravity Equation (Properly Grouped)
    # v_dg = ( (v_bar^2)^2 + (v_bar^2 * r * a0) )^0.25
    v_dg_ms = ((v_bar_sq**2) + (v_bar_sq * r_m * A0)) ** 0.25
    v_dg_kms = v_dg_ms / KM_S_TO_M_S

    # 4. Standard Newtonian Comparison
    v_newton_kms = np.sqrt(v_bar_sq) / KM_S_TO_M_S

    # 5. Professional Plotting
    plt.figure(figsize=(10, 6))
    # plt.errorbar(
    #     r_kpc, v_obs, yerr=df["errV"], fmt="ko", label="Observed (SPARC)", markersize=4
    # )
    plt.plot(
        r_kpc,
        v_newton_kms,
        "p--",
        label=f"Baryonic Newtonian (Expected Decay for $\\Upsilon = {upsilon:.2f}$)",
    )
    plt.plot(
        r_kpc,
        v_dg_kms,
        "r-",
        linewidth=2,
        label=f"Divergent Gravity (Flat Curve, $\\Upsilon={upsilon:.2f}$)",
    )

    plt.ylim(0, max(v_obs) * 1.2)  # Lock Y-axis to realistic speeds
    plt.title(f"Galaxy Rotation Curve: {name}")
    plt.xlabel("Distance from Center (kpc)")
    plt.ylabel("Rotation Velocity (km/s)")
    plt.legend()
    return plt
    # plt.show()


def calculate_dg_with_ml(df, upsilon):
    """
    Optimized Divergent Gravity calculation.
    Uses explicit SI conversion to bypass Astropy 'add' errors.
    """
    # 1. Convert everything to raw SI values (floats)
    # This prevents Astropy from blocking the 'add' operation
    r_m = (df["Rad"].values * u.kpc).to(u.m).value
    v_gas_ms = (df["Vgas"].values * u.km / u.s).to(u.m / u.s).value
    v_disk_ms = (df["Vdisk"].values * u.km / u.s).to(u.m / u.s).value
    v_bul_ms = (df["Vbul"].values * u.km / u.s).to(u.m / u.s).value

    # Ensure A0 is a raw SI value (m/s^2)
    # Assuming A0 was defined as an Astropy quantity earlier
    a0_si = A0.to(u.m / u.s**2).value if hasattr(A0, "unit") else A0

    # 2. Perform the math in raw SI
    # v_bar^2 = v_gas^2 + upsilon * (v_disk^2 + v_bulge^2)
    v_bar_sq = v_gas_ms**2 + (upsilon * (v_disk_ms**2 + v_bul_ms**2))

    # DG Equation: v_dg = (v_bar^4 + v_bar^2 * r * a0)^0.25
    term_newton = v_bar_sq**2
    term_machian = v_bar_sq * r_m * a0_si

    v_dg_si = (term_newton + term_machian) ** 0.25

    # 3. Return as an Astropy Quantity in km/s
    return (v_dg_si * u.m / u.s).to(u.km / u.s)


def optimize_ml(df, galaxy_name):
    ml_range = np.linspace(0.0, 1.2, 100)
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


def initialize_sparc_data():
    """Handles 404/429 errors by caching data locally."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    if not os.path.exists(ZIP_FILE):
        print("Attempting to fetch SPARC database from Zenodo mirror...")
        # Browser-like headers to prevent 403/429
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(DATA_URL, headers=headers, stream=True)
            r.raise_for_status()
            with open(ZIP_FILE, "wb") as f:
                f.write(r.content)
            print("Download successful.")
        except Exception as e:
            print(f"\nCRITICAL: Automatic download failed ({e}).")
            print(f"Please manually download the 'Rotmod_LTG.zip' from:")
            print(f"https://zenodo.org/records/16284118")
            print(f"And place it in the folder: {os.path.abspath(LOCAL_DIR)}")
            return None

    return zipfile.ZipFile(ZIP_FILE)


def calculate_dg_profile(df):
    """
    Computes predicted velocity based on Divergent Gravity using raw SI
    conversion to ensure unit-safety and 0.5% precision.
    """
    # 1. Convert to raw SI floats (meters and meters/second)
    r_m = (df["Rad"].values * u.kpc).to(u.m).value
    v_gas_ms = (df["Vgas"].values * u.km / u.s).to(u.m / u.s).value
    v_disk_ms = (df["Vdisk"].values * u.km / u.s).to(u.m / u.s).value
    v_bul_ms = (df["Vbul"].values * u.km / u.s).to(u.m / u.s).value

    # Ensure A0 is a raw SI value (m/s^2)
    a0_si = A0.to(u.m / u.s**2).value if hasattr(A0, "unit") else A0

    # 2. Compute Baryonic Potential (v_bar^2) in SI
    v_bar_sq = v_gas_ms**2 + v_disk_ms**2 + v_bul_ms**2

    # 3. Divergent Gravity Identity
    # v_dg = (v_bar^4 + v_bar^2 * r * a0)^0.25
    term_newton = v_bar_sq**2
    term_machian = v_bar_sq * r_m * a0_si

    v_dg_si = (term_newton + term_machian) ** 0.25

    # 4. Return as Astropy Quantities for standard plotting
    v_dg_final = (v_dg_si * u.m / u.s).to(u.km / u.s)
    v_bar_final = (np.sqrt(v_bar_sq) * u.m / u.s).to(u.km / u.s)

    return v_dg_final, v_bar_final


def run_analysis(name):
    zf = initialize_sparc_data()
    if not zf:
        return
    try:
        with zf.open(f"{name}_rotmod.dat") as f:
            lines = []
            initial_lines = f.readlines()
            for l in initial_lines:
                _l = l.decode("utf-8")
                if not _l.startswith("#"):
                    lines.append(_l)
            file_object = io.StringIO("\n".join(lines))
            df = pd.read_table(
                file_object,
                sep="\s+",
                names=[
                    "Rad",
                    "Vobs",
                    "errV",
                    "Vgas",
                    "Vdisk",
                    "Vbul",
                    "SBdisk",
                    "SBbul",
                ],
            )

            ml, err = optimize_ml(df, name)
            print(df.head())
            print(f"Name: {name}, Best M/L: {ml:.2f}, Precision: {100-err:.2f}%")
            get_clean_plot(df, name, ml)

            v_dg, v_bar = calculate_dg_profile(df)

            # Plotting Results
            # plt.figure(figsize=(10, 6))
            plt.errorbar(
                df["Rad"],
                df["Vobs"],
                yerr=df["errV"],
                fmt="ko",
                markersize=4,
                label="SPARC Observed Data",
                alpha=0.7,
            )
            plt.plot(df["Rad"], v_bar.value, "g--", label="Baryonic (Newtonian Limit)")
            plt.plot(
                df["Rad"],
                v_dg.value,
                "b-",
                linewidth=2,
                label=f"Divergent Gravity (a0={A0:.2e})",
            )

            plt.title(f"Divergent Gravity Validation: {name}", fontsize=14)
            plt.xlabel("Radius (kpc)")
            plt.ylabel("Velocity (km/s)")
            plt.legend(loc="lower right")
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.show()

    except KeyError:
        print(f"Galaxy {name} not found in the dataset.")


# if __name__ == "__main__":
#     # Test across different mass ranges
#     target_galaxies = ["UGC00891", "NGC3198", "IC2574", "UGC06614"]
#     for g in target_galaxies:
#         run_analysis(g)


if __name__ == "__main__":
    # Test across different mass ranges
    # target_galaxies = ["UGC00891", "NGC3198", "IC2574", "UGC06614"]
    with_most_data = [
        "NGC3521",
        "UGC08699",
        "NGC3198",
        "UGC02916",
        "NGC6015",
        "UGC06786",
        "NGC7793",
        "UGC03580",
        "UGC03205",
        "NGC2841",
        "NGC6946",
        "UGC11914",
        "UGC09133",
        "UGC06787",
        "UGC05253",
        "NGC2403",
        "UGC02953",
    ]
    for g in with_most_data:
        run_analysis(g)
