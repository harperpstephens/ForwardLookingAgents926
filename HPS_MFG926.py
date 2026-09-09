import getpass
import sys

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# 1. Acquiring Empirical Data

# Codes Census Bureau uses for missing values.
MISSING_CODES = {-66666666, -222222222, -333333333, -555555555,
                 -666666666, -888888888, -999999999}

def getcensus_migration_data(api_key, state_fips="39", county_fips="089"):
    """
    These pulls the county-level migration data for Lick County (site), Ohio (state)
    while converting total migration flow into a friction parameter (c_friction).

    You will be asked to provide your own unique API key, which can be provided for
    free from the U.S. Census Bureau API at https://api.census.gov/data/key_signup.html
    """
    print("Getting county migration data for Lick County (site)")
    url = "https://api.census.gov/data/2020/acs/flows"
    params = {
        "get": "MOVEDIN,MOVEDOUT,MOVEDNET,GEOID1,GEOID2,FULL1_NAME,FULL2_NAME",
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
        "key": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout = 30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed contacting Census API: {e}")
    if not response.ok:
        raise RuntimeError(
            f"Request failed for Census API: [{response.status_code}]: "
            f"{response.text[:300]}"
        )

    data = response.json()
    if len(data) < 2:
        raise RuntimeError(
            "Census API did not return any results for "
            f"(county:{county_fips}, state:{state_fips})."
        )
    df = pd.DataFrame(data[1:], columns=data[0])
    df[["MOVEDIN", "MOVEDOUT"]] = df[["MOVEDIN", "MOVEDOUT"]].apply(
        pd.to_numeric, errors="coerce"
    )
    mask = ~df["MOVEDIN"].isin(MISSING_CODES) & ~df["MOVEDOUT"].isin(MISSING_CODES)
    dfclean = df[mask].dropna(subset=["MOVEDIN", "MOVEDOUT"])
    if dfclean.empty:
        raise RuntimeError(
            "All rows were missing after filtering. No usable flow data for this area."
        )

    flowvolume = dfclean["MOVEDIN"].sum() + dfclean["MOVEDOUT"].sum()
    print(f"Data retrieved with {len(dfclean)} valid rows out of "
          f"{len(df)}, total flows = {flowvolume:,.0f}")
    if flowvolume <= 0:
        raise RuntimeError("Flow volume was zero or negative.")
    return max(0.2, min(1.5, 100000 / (flowvolume + 1)))

def get_apikey():
    key = getpass.getpass("Enter your Census API key: ").strip()
    if not key:
        print("No API key provided.", file=sys.stderr)
        sys.exit(1)
    return key

if __name__ == "__main__":
    api_key = get_apikey()
    try:
        c_friction = getcensus_migration_data(api_key)
    except RuntimeError as e:
        print(f"Could not retrieve ACS data - {e}",
              file=sys.stderr)
        print("Stopping, no fallback values will be used for analysis.", file=sys.stderr)
        sys.exit(1)
    print(f"Empirical Friction = {c_friction:.4f}\n")

    # 2. Solving the Mean Field Game (CFL-Stabilized).
    print("Beginning to solve the MFG.")

    nx = 100
    x = np.linspace(-1,1,nx)
    dx = x[1] - x[0]

    T_final = 8.0 # Years
    sigma_squared = 0.05
    max_speed = 5.0

    v_max_empirical = max_speed / c_friction
    dt_diffusion = (dx ** 2) / (2.0 * sigma_squared)
    dt_advection = dx / v_max_empirical
    max_safe_dt = min(dt_diffusion, dt_advection) * 0.9

    nt = int(T_final / max_safe_dt) + 1
    dt = T_final / nt
    t_array = np.linspace(0, T_final, nt)
    print(f"CFL stabilized: Grid requires nt={nt} with time steps (dt={dt:.5f})")

    kappa = 1.0
    gamma = 0.3
    T_prod = 3.0

    def wage(xpos, t):
        base_wage = 2.0
        if t >= T_prod:
            return base_wage + 3.5 * np.exp(-(xpos ** 2) / 0.02)
        return base_wage

    L = np.ones((nt, nx)) / 2.0
    V = np.zeros((nt, nx))

    iterations = 8
    for step in range(iterations):
        print(f"Iteration {step + 1}/{iterations}: Forward-Backward.")
        L_old = L.copy()
        L_new = np.zeros_like(L)
        L_new[0, :] = L[0, :]

        # HJB Sweep (Backward)
        V[-1, :] = wage(x, T_final) - kappa * (L_old[-1, :] ** gamma)
        for n in range(nt - 2, -1, -1):
            V_curr = V[n + 1, :]

            dV_f, dV_b = np.zeros(nx), np.zeros(nx)
            dV_f[:-1] = (V_curr[1:] - V_curr[:-1]) / dx
            dV_b[1:] = (V_curr[1:] - V_curr[:-1]) / dx

            dV_f = np.clip(dV_f, -max_speed, max_speed)
            dV_b = np.clip(dV_b, -max_speed, max_speed)

            H = (1 / (2 * c_friction)) * (np.maximum(dV_f, 0) ** 2 + np.minimum(dV_b, 0) ** 2)

            d2V = np.zeros(nx)
            d2V[1:-1] = (V_curr[2:] - 2 * V_curr[1:-1] + V_curr[:-2]) / (dx ** 2)
            d2V[0] = 2 * (V_curr[1] - V_curr[0]) / (dx ** 2)
            d2V[-1] = 2 * (V_curr[-2] - V_curr[-1]) / (dx ** 2)

            W = wage(x, t_array[n])

            V[n, :] = V_curr + dt * (0.5 * sigma_squared * d2V - H + W - kappa * (L_old[n, :] ** gamma))

        # KFE Sweep (Forward).
        for n in range(0, nt - 1):
            L_curr = L_new[n, :]

            dV_c = np.zeros(nx)
            dV_c[1:-1] = (V[n, 2:] - V[n, :-2]) / (2 * dx)
            dV_c = np.clip(dV_c, -max_speed, max_speed)
            v = dV_c / c_friction

            v_inter = (v[1:] + v[:-1]) / 2.0
            L_inter = (L_curr[1:] + L_curr[:-1]) / 2.0
            flux_inter = v_inter * L_inter - (sigma_squared / 2.0) * (L_curr[1:] - L_curr[:-1]) / dx

            F = np.zeros(nx + 1)
            F[1:-1] = flux_inter

            L_new[n + 1, :] = L_curr - (dt / dx) * (F[1:] - F[:-1])
            L_new[n + 1, :] = np.maximum(L_new[n + 1, :], 1e-8)
            L_new[n + 1, :] = L_new[n + 1, :] / (np.sum(L_new[n + 1, :]) * dx)

        L = 0.2 * L_new + 0.8 * L_old

    # 3. Output
    print("\nExporting visual.")

    plt.figure(figsize=(8, 5))
    plt.plot(x, L[0, :], color="grey", linestyle="--", label="t=0 (Initial)")
    plt.plot(x, L[int(nt * 0.35), :], color="blue", label=f"t={t_array[int(nt * 0.35)]:.1f} (Pre-Shock Anticipation)")
    plt.plot(x, L[-1, :], color="green", label=f"t={T_final:.1f} (Post-Shock Equilibrium)")
    plt.axvline(0, color="red", linestyle=":", label="x=0 (Intel Site)")
    plt.title("Anticipatory Spatial Reallocation (CHIPS Act Shock)")
    plt.xlabel("Spatial Grid (x)")
    plt.ylabel("Population Density (L(x,t))")
    plt.legend()
    plt.tight_layout()
    plt.savefig("HPS_MFG926.png", dpi=300)
    print("Saved visual.")

print("Done.")