import getpass
import sys
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# 1. Pulling Empirical Data.
MISSING_CODES = {-66666666, -222222222, -333333333, -555555555,
                 -666666666, -888888888, -999999999}
targetcounty = {"state": "39", "county": "089", "name": "Licking County"}
adjacentcounties = [
    {"state": "39", "county": "041", "name": "Delaware County"},
    {"state": "39", "county": "045", "name": "Fairfield County"},
    {"state": "39", "county": "049", "name": "Franklin County"},
    {"state": "39", "county": "083", "name": "Knox County"},
    {"state": "39", "county": "119", "name": "Muskingum County"},
    {"state": "39", "county": "115", "name": "Perry County"},
]


def pullmigration_flowsums(apikey, statefips, countyfips):
    """
    These pulls the county-level migration data for Lick County (site), Ohio (state)
    while converting total migration flow into a friction parameter (c_friction).
    
    You will be asked to provide your own unique API key, which can be provided for
    free from the U.S. Census Bureau API at https://api.census.gov/data/key_signup.html
    """
    url = "https://api.census.gov/data/2020/acs/flows"
    params = {
        "get": "MOVEDIN,MOVEDOUT,MOVEDNET,GEOID1,GEOID2,FULL1_NAME,FULL2_NAME",
        "for": f"county:{countyfips}",
        "in": f"state:{statefips}",
        "key": apikey,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e
    if not response.ok:
        raise RuntimeError(f"Request failed: {response.status_code}")

    data = response.json()
    if len(data) < 2:
        raise RuntimeError(f"Census API returned no data (state: {statefips}, county: {countyfips}).")

    df = pd.DataFrame(data[1:], columns=data[0])
    df[["MOVEDIN", "MOVEDOUT"]] = df[["MOVEDIN", "MOVEDOUT"]].apply(pd.to_numeric, errors="coerce")
    mask = ~df["MOVEDIN"].isin(MISSING_CODES) & ~df["MOVEDOUT"].isin(MISSING_CODES)
    df_clean = df[mask].dropna(subset=["MOVEDIN", "MOVEDOUT"])
    if df_clean.empty:
        raise RuntimeError("All rows missing after filtering codes.")

    return df_clean["MOVEDIN"].sum(), df_clean["MOVEDOUT"].sum(), len(df_clean), len(df)


def pullcensus_migrationdata(apikey, statefips="39", countyfips="089"):
    moved_in, moved_out, n_valid, n_total = pullmigration_flowsums(apikey, statefips, countyfips)
    flowvolume = moved_in + moved_out
    print(f"Retrieved {n_valid}/{n_total} valid flow records. Total flows = {flowvolume:,.0f}")
    if flowvolume <= 0:
        raise RuntimeError("Flow volume cannot be negative or zero.")
    return max(0.2, min(1.5, 100000 / (flowvolume + 1)))


def pullcounty_population(apikey, statefips, countyfips):
    url = "https://api.census.gov/data/2020/acs/acs5"
    params = {"get": "B01003_001E,NAME", "for": f"county:{countyfips}", "in": f"state:{statefips}", "key": apikey}
    response = requests.get(url, params=params, timeout=30)
    if not response.ok: raise RuntimeError("Census API request failed")
    data = response.json()
    pop = pd.to_numeric(dict(zip(data[0], data[1]))["B01003_001E"], errors="coerce")
    return float(pop)


def pullcounty_medianearnings(apikey, statefips, countyfips):
    url = "https://api.census.gov/data/2020/acs/acs5/subject"
    params = {"get": "S2001_C01_002E,NAME", "for": f"county:{countyfips}", "in": f"state:{statefips}", "key": apikey}
    response = requests.get(url, params=params, timeout=30)
    if not response.ok: raise RuntimeError("Census API request failed")
    data = response.json()
    wage = pd.to_numeric(dict(zip(data[0], data[1]))["S2001_C01_002E"], errors="coerce")
    return float(wage)


def buildsummary_statisticstable(apikey, c_val):
    print("Building summary statistics table.")
    counties = [targetcounty] + adjacentcounties
    records = {}
    for county in counties:
        pop = pullcounty_population(apikey, county["state"], county["county"])
        moved_in, moved_out, _, _ = pullmigration_flowsums(apikey, county["state"], county["county"])
        wage = pullcounty_medianearnings(apikey, county["state"], county["county"])
        records[county["name"]] = {"population": pop, "moved_in": moved_in, "moved_out": moved_out, "wage": wage}

    target = records[targetcounty["name"]]
    adjacent = [records[county["name"]] for county in adjacentcounties]
    allcounties = [target] + adjacent

    def weighted_wage(rows):
        totalpop = sum(row["population"] for row in rows)
        return sum(row["wage"] * row["population"] for row in rows) / totalpop

    table = pd.DataFrame([
        {"Metric": "Base Population", "Licking County": f"{target['population']:,.0f}",
         "Adjacent Counties": f"{sum(r['population'] for r in adjacent):,.0f}",
         "Commuting Zone Total": f"{sum(r['population'] for r in allcounties):,.0f}"},
        {"Metric": "Gross In-Migration (Annual)", "Licking County": f"{target['moved_in']:,.0f}",
         "Adjacent Counties": f"{sum(r['moved_in'] for r in adjacent):,.0f}",
         "Commuting Zone Total": f"{sum(r['moved_in'] for r in allcounties):,.0f}"},
        {"Metric": "Gross Out-Migration (Annual)", "Licking County": f"{target['moved_out']:,.0f}",
         "Adjacent Counties": f"{sum(r['moved_out'] for r in adjacent):,.0f}",
         "Commuting Zone Total": f"{sum(r['moved_out'] for r in allcounties):,.0f}"},
        {"Metric": "Baseline Median Wage", "Licking County": f"${target['wage']:,.0f}",
         "Adjacent Counties": f"${weighted_wage(adjacent):,.0f}",
         "Commuting Zone Total": f"${weighted_wage(allcounties):,.0f}"},
        {"Metric": "Derived Moving Friction (c)", "Licking County": "--", "Adjacent Counties": "--",
         "Commuting Zone Total": f"{c_val:.2f}"},
    ])
    return table


# 2. Mean Field Game Solver
def solve_mfg(c_friction, label):
    nx = 100
    x = np.linspace(-1, 1, nx)
    dx = x[1] - x[0]

    # Increased sigma_squared term to 0.15 for realistic preference dispersion.
    T_final, sigma_squared, max_speed = 8.0, 0.15, 5.0

    alpha_relax = max(0.015, min(0.2, c_friction * 0.15))
    iterations = 35 if c_friction < 0.5 else 15

    v_max_cfl = max_speed / c_friction
    max_safe_dt = min((dx ** 2) / (2.0 * sigma_squared), dx / v_max_cfl) * 0.25
    nt = int(T_final / max_safe_dt) + 1
    dt, t_array = T_final / nt, np.linspace(0, T_final, nt)

    kappa, gamma, lambda_rate = 1.0, 0.3, 0.33

    # Linear Commute Gradient implementation.
    W_base = np.full(nx, 2.0)
    W_premium = 0.5
    tau = 1.2
    W_post = W_base + np.maximum(0, W_premium - tau * np.abs(x))

    L = np.ones((nt, nx)) / 2.0
    V_pre = np.zeros((nt, nx))
    V_post = np.zeros((nt, nx))

    for step in range(iterations):
        L_old, L_new = L.copy(), np.zeros_like(L)
        L_new[0, :] = L[0, :]

        # HJB Sweep Post-Shock.
        V_post[-1, :] = W_post - kappa * (L_old[-1, :] ** gamma)
        for n in range(nt - 2, -1, -1):
            V_curr = V_post[n + 1, :]
            dV_f, dV_b = np.zeros(nx), np.zeros(nx)
            dV_f[:-1] = (V_curr[1:] - V_curr[:-1]) / dx
            dV_b[1:] = (V_curr[1:] - V_curr[:-1]) / dx

            dV_f = np.clip(dV_f, -max_speed, max_speed)
            dV_b = np.clip(dV_b, -max_speed, max_speed)

            H = (1.0 / (2.0 * c_friction)) * (np.maximum(dV_b, 0.0) ** 2 + np.minimum(dV_f, 0.0) ** 2)

            d2V = np.zeros(nx)
            d2V[1:-1] = (V_curr[2:] - 2 * V_curr[1:-1] + V_curr[:-2]) / (dx ** 2)
            d2V[0] = 2 * (V_curr[1] - V_curr[0]) / (dx ** 2)
            d2V[-1] = 2 * (V_curr[-2] - V_curr[-1]) / (dx ** 2)

            V_post[n, :] = V_curr + dt * (0.5 * sigma_squared * d2V + H + W_post - kappa * (L_old[n, :] ** gamma))

        # HJB Sweep Pre-Shock.
        V_pre[-1, :] = W_base - kappa * (L_old[-1, :] ** gamma)
        for n in range(nt - 2, -1, -1):
            V_curr = V_pre[n + 1, :]
            dV_f, dV_b = np.zeros(nx), np.zeros(nx)
            dV_f[:-1] = (V_curr[1:] - V_curr[:-1]) / dx
            dV_b[1:] = (V_curr[1:] - V_curr[:-1]) / dx

            dV_f = np.clip(dV_f, -max_speed, max_speed)
            dV_b = np.clip(dV_b, -max_speed, max_speed)

            H = (1.0 / (2.0 * c_friction)) * (np.maximum(dV_b, 0.0) ** 2 + np.minimum(dV_f, 0.0) ** 2)

            d2V = np.zeros(nx)
            d2V[1:-1] = (V_curr[2:] - 2 * V_curr[1:-1] + V_curr[:-2]) / (dx ** 2)
            d2V[0] = 2 * (V_curr[1] - V_curr[0]) / (dx ** 2)
            d2V[-1] = 2 * (V_curr[-2] - V_curr[-1]) / (dx ** 2)

            jump_term = lambda_rate * (V_post[n + 1, :] - V_curr)
            V_pre[n, :] = V_curr + dt * (
                    0.5 * sigma_squared * d2V + H + W_base - kappa * (L_old[n, :] ** gamma) + jump_term)

        # KFE Sweep Forward (Aligned Upwind).
        for n in range(0, nt - 1):
            dV_f, dV_b = np.zeros(nx), np.zeros(nx)
            dV_f[:-1] = (V_pre[n, 1:] - V_pre[n, :-1]) / dx
            dV_b[1:] = (V_pre[n, 1:] - V_pre[n, :-1]) / dx

            v_b = np.maximum(np.clip(dV_b, -max_speed, max_speed), 0.0) / c_friction
            v_f = np.minimum(np.clip(dV_f, -max_speed, max_speed), 0.0) / c_friction
            v = v_b + v_f

            v_inter = (v[1:] + v[:-1]) / 2.0
            v_pos = np.maximum(v_inter, 0.0)
            v_neg = np.minimum(v_inter, 0.0)

            flux_adv = v_pos * L_new[n, :-1] + v_neg * L_new[n, 1:]
            flux_diff = -(sigma_squared / 2.0) * (L_new[n, 1:] - L_new[n, :-1]) / dx

            F = np.zeros(nx + 1)
            F[1:-1] = flux_adv + flux_diff

            L_new[n + 1, :] = L_new[n, :] - (dt / dx) * (F[1:] - F[:-1])
            L_new[n + 1, :] = np.maximum(L_new[n + 1, :], 1e-12)
            L_new[n + 1, :] /= np.sum(L_new[n + 1, :]) * dx

        L = alpha_relax * L_new + (1.0 - alpha_relax) * L_old

    welfare = np.sum(L[-1, :] * V_pre[-1, :]) * dx
    return x, t_array, L, V_pre, welfare


if __name__ == "__main__":
    apikey = getpass.getpass("Enter your Census API key: ").strip()
    if not apikey: sys.exit("No API key provided.")

    try:
        c_empirical = pullcensus_migrationdata(apikey)
        summarytable = buildsummary_statisticstable(apikey, c_empirical)
        print("\nSummary Statistics:\n", summarytable.to_string(index=False))
        summarytable.to_csv("summary_statisticstable.csv", index=False)
    except RuntimeError as e:
        sys.exit(f"Data retrieval failed: {e}")

    print(f"\nSolving Empirical Model (c={c_empirical:.2f}).")
    x, t, L_emp, V_emp, wf_emp = solve_mfg(c_empirical, "Empirical")
    print(f"Aggregate Welfare (Empirical): {wf_emp:.4f}")

    print("Solving Baseline Model (c=0.1).")
    _, _, L_base, _, wf_base = solve_mfg(0.1, "Baseline")

    print("Solving Sensitivity (c=1.0).")
    _, _, L_s1, _, wf_s1 = solve_mfg(1.0, "c=1.0")

    print("Solving Sensitivity (c=2.0).")
    _, _, L_s2, _, wf_s2 = solve_mfg(2.0, "c=2.0")

    # Output Plot.
    plt.figure(figsize=(9, 6))
    plt.plot(x, L_emp[0, :], color="grey", linestyle="--", label="t=0 (Initial)")
    plt.plot(x, L_base[-1, :], color="orange", linestyle="-.", label="t=8.0 (Baseline c=0.1)")
    plt.plot(x, L_emp[int(len(t) * 0.35), :], color="blue", label="t=2.8 (Empirical Pre-Shock)")
    plt.plot(x, L_emp[-1, :], color="green", label=f"t=8.0 (Empirical Post-Shock c={c_empirical:.2f})")

    plt.axvline(0, color="red", linestyle=":", label="x=0 (Intel Site)")
    plt.title("Anticipatory Spatial Reallocation (CHIPS Act Shock)")
    plt.xlabel("Spatial Grid (x)")
    plt.ylabel("Population Density (L(x,t))")
    plt.legend()
    plt.tight_layout()
    plt.savefig("HPS_MFG926.png", dpi=300)
    print("Exported HPS_MFG926.png.")
