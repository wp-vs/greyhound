"""Romford predictions using time-distribution Monte Carlo simulation.

For each dog:
  1. Estimate their finishing time distribution N(mu, sigma) from recent runs
     at similar distances, with decay weighting for recency
  2. Adjust for track/trap effects
  3. Monte Carlo simulate 50,000 races — each dog draws a random time
  4. Win probability = fraction of simulations where dog has the lowest time

This produces much more realistic probabilities than z-score/softmax approaches
because it naturally captures the overlap between dogs' time distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
np.random.seed(42)

N_SIMS = 50_000

# Odds from Oddschecker (user-provided)
odds_data = {
    1: {"Goulane Vegas": "5/2", "Bonville Secret": "11/4", "Droopys Rhona": "10/3",
        "Bluejig Designer": "4", "Abel Mabel": "9/2", "Olwinn Seen It": "7"},
    2: {"Calypso Blaze": "5/4", "Lil Bo Beep": "10/3", "Headford Wayne": "5",
        "Aero Espresso": "6", "Skidroe Lady": "7", "Hellofalooker": "14"},
    3: {"Rockmount Kellie": "7/4", "Mahoonagh Hoffa": "10/3", "Bretons Girl": "7/2",
        "My Lil Bella": "6", "Brindle Moon": "8", "Jayar Rogue": "9"},
    4: {"Bretons Boy": "3", "Lesleys Buddy": "3", "Stradeen Spirit": "10/3",
        "Alan The First": "9/2", "Piemans Fletch": "9/2", "Bombay Trend": "8"},
    5: {"Aayamza Legend": "3", "Farneys Tearaway": "10/3", "Droopys Rosie": "7/2",
        "Ballymac Florie": "9/2", "Miami Yeats": "11/2", "Droopys Maybe": "13/2"},
    6: {"Slingshot Poppy": "1/2", "Flashing Fender": "10/3", "Da Don": "6",
        "Yahoo Mareike": "14", "Droopys Will": "20", "Broadway Steel": "33"},
    7: {"Bockos Buster": "11/18", "Jacktavern Don": "5", "Sehnsa Amigo": "6",
        "Quinton Boy": "7", "Uncle Ed": "14", "Charlie Loves Me": "16"},
    8: {"Piemans Goalie": "13/8", "Swift Hostile": "28/15", "Pro Parker": "5",
        "Bombay Buck": "7", "Funky Adz": "8", "Tally Ho Socks": "20"},
    9: {"Unmistakeable": "1", "Alana The Second": "11/4", "Cooladerrydancer": "6",
        "Essjay Sonia": "13/2", "Letter Cutie": "14", "Makeit Poppy": "14"},
    10: {"Underground Jim": "6/4", "Aero Leg It": "3", "Thorpys Legacy": "4",
         "Ivanhoe Rose": "13/2", "Ballymac Suntan": "7", "Real Gone Lover": "10"},
    11: {"Out The Blue": "2", "Royal Hotshot": "2", "Bacon Frazzles": "4",
         "Inca Lewie": "8", "Zenith Angel": "8", "Alans Amigo": "14"},
    12: {"Princess Matilda": "5/6", "Intriguing Iris": "5/2", "Innfield Fifi": "9/2",
         "Full Monty": "16", "Getaway Car": "16", "Swift Agile": "16"},
}


def parse_odds(odds_str):
    odds_str = str(odds_str).strip()
    if "/" in odds_str:
        parts = odds_str.split("/")
        return float(parts[0]) / float(parts[1]) + 1
    else:
        return float(odds_str) + 1


# Load data
rc = pd.read_csv(project_root / "data/romford_racecard_full_2026-03-13.csv")
history = pd.read_csv(project_root / "data/romford_combined_history_2026-03-13.csv")
rc["greyhound"] = rc["greyhound"].str.strip().str.title()
history["greyhound"] = history["greyhound"].str.strip().str.title()

# Exclude today's entries
history = history[history["date"] < "2026-03-13"].copy()
print(f"History (excl today): {len(history)} records for {history['greyhound'].nunique()} dogs")

# Pre-compute population sectional time stats by TRACK + DISTANCE (for early speed factor)
# Sectional split points differ by track, so cross-track comparison is meaningless.
# e.g. Hove 500m sectional = 3.66s, Towcester 500m = 4.21s, Nottingham 500m = 5.21s
_sectional_stats = {}  # keyed by (track, distance)
if "sectional_time" in history.columns:
    sect_valid = history[history["sectional_time"].notna() & (history["sectional_time"] > 0)]
    for (track, dist), group in sect_valid.groupby(["track", "distance"]):
        sub = group["sectional_time"]
        if len(sub) >= 10:
            _sectional_stats[(track.lower().strip(), dist)] = (sub.mean(), sub.std())

# Add odds
rc["odds_frac"] = ""
rc["odds_dec"] = np.nan
rc["mkt_prob_raw"] = np.nan
rc["mkt_prob"] = np.nan

for race_num, odds in odds_data.items():
    race_mask = rc["race_number"] == race_num
    total_imp = 0
    for dog, frac in odds.items():
        dec = parse_odds(frac)
        imp = 1.0 / dec
        total_imp += imp
        mask = race_mask & (rc["greyhound"] == dog.strip().title())
        rc.loc[mask, "odds_frac"] = frac
        rc.loc[mask, "odds_dec"] = round(dec, 2)
        rc.loc[mask, "mkt_prob_raw"] = round(imp, 4)
    if total_imp > 0:
        for dog, frac in odds.items():
            mask = race_mask & (rc["greyhound"] == dog.strip().title())
            raw = rc.loc[mask, "mkt_prob_raw"].values
            if len(raw) > 0:
                rc.loc[mask, "mkt_prob"] = round(raw[0] / total_imp, 4)


# Romford trap bias in seconds (calibrated from 1481 Romford runs)
# Win rates: T1=38.2%, T2=23.1%, T3=31.6%, T4=27.5%, T5=23.2%, T6=23.0%
# Fair baseline = 16.7% (1/6). Convert win rate advantage to time adjustment.
# Larger magnitude than before — data shows T1 has a massive rail advantage.
# Positive = slower, negative = faster
TRAP_BIAS_SECS = {1: -0.12, 2: 0.02, 3: -0.06, 4: -0.02, 5: 0.03, 6: 0.05}

# Race gap (days since last race) adjustment in seconds
# Data shows: 0-3d baseline, 4-9d similar, 30-59d faster (40% WR), 60d+ faster (48% WR)
# Long-rested dogs seem to perform well — likely returning from targeted rest, not injury.
# We apply a small bonus for 30-60d gaps and neutral for very long gaps (high uncertainty).
RACE_GAP_ADJUSTMENT = {
    (0, 6): 0.0,       # normal scheduling, baseline
    (7, 13): 0.0,      # weekly, baseline
    (14, 29): 0.02,    # slightly rusty if 2-4 weeks off
    (30, 59): -0.06,   # data shows strong performance after deliberate rest
    (60, 999): -0.03,  # still positive signal but more uncertain (small sample)
}

# Early speed / sectional time factor
# Fast starters win 33.8% vs 15.6% for slow starters at Romford
# We model this as a time adjustment based on how a dog's sectional compares
# to the distance-average sectional time

# Typical std dev for greyhound times when we have no data
# Greyhound races are quite variable — typical spread ~0.3-0.5s at 400m
DEFAULT_STD_BY_DIST = {225: 0.25, 264: 0.28, 277: 0.28, 400: 0.35, 462: 0.38,
                       480: 0.40, 575: 0.45, 750: 0.55}


def get_default_std(dist):
    """Get a reasonable default std for a given distance."""
    if dist in DEFAULT_STD_BY_DIST:
        return DEFAULT_STD_BY_DIST[dist]
    # Interpolate
    dists = sorted(DEFAULT_STD_BY_DIST.keys())
    if dist <= dists[0]:
        return DEFAULT_STD_BY_DIST[dists[0]]
    if dist >= dists[-1]:
        return DEFAULT_STD_BY_DIST[dists[-1]]
    for i in range(len(dists) - 1):
        if dists[i] <= dist <= dists[i + 1]:
            frac = (dist - dists[i]) / (dists[i + 1] - dists[i])
            return DEFAULT_STD_BY_DIST[dists[i]] * (1 - frac) + DEFAULT_STD_BY_DIST[dists[i + 1]] * frac
    return 0.35


def get_race_gap_adjustment(hist, race_date="2026-03-13"):
    """Compute time adjustment based on days since last race."""
    if hist.empty:
        return 0.0, None
    dates = pd.to_datetime(hist["date"], format="mixed")
    last_race = dates.max()
    gap_days = (pd.Timestamp(race_date) - last_race).days
    for (lo, hi), adj in RACE_GAP_ADJUSTMENT.items():
        if lo <= gap_days <= hi:
            return adj, gap_days
    return 0.0, gap_days


def get_early_speed_adjustment(hist, race_dist, race_track):
    """Compute time adjustment based on dog's early speed (sectional time).

    Compares the dog's average sectional time to the population average
    for the SAME TRACK and SAME DISTANCE. Sectional split points differ by track
    (e.g. Hove 500m = 3.66s, Towcester 500m = 4.21s) so cross-track comparison
    is meaningless.
    """
    if hist.empty or "sectional_time" not in hist.columns:
        return 0.0, np.nan

    # Only use runs at the SAME TRACK and SAME DISTANCE (±30m)
    has_sect = hist["sectional_time"].notna() & (hist["sectional_time"] > 0)
    same_dist = (hist["distance"] - race_dist).abs() <= 30
    if "track" in hist.columns:
        same_track = hist["track"].str.lower().str.strip() == race_track.lower().strip()
        relevant = hist[has_sect & same_dist & same_track]
    else:
        relevant = hist[has_sect & same_dist]

    if len(relevant) < 2:
        return 0.0, np.nan

    dog_sect = relevant["sectional_time"].mean()
    # Population sectional mean for this track + distance
    pop_sect = _sectional_stats.get((race_track.lower().strip(), race_dist), None)
    if pop_sect is None:
        return 0.0, dog_sect

    pop_mean, pop_std = pop_sect
    if pop_std <= 0:
        return 0.0, dog_sect

    # How many SDs faster/slower than average?
    # Negative z = faster sectional = better early speed
    z = (dog_sect - pop_mean) / pop_std

    # Convert to time adjustment: fast starters are genuinely faster overall
    # ~0.08s per SD of early speed advantage (calibrated to match 33.8% vs 15.6% win rates)
    adjustment = z * 0.08
    return adjustment, dog_sect


def estimate_time_distribution(name, hist, race_dist, race_track):
    """Estimate a dog's finishing time distribution N(mu, sigma).

    Returns dict with est_mu, est_sigma, n_timed, n_runs, n_wins, win_rate, recent_form,
    race_gap_days, sectional_avg, gap_adj, speed_adj.
    If no timed runs at similar distance, uses race-level defaults.
    """
    result = {
        "est_mu": np.nan, "est_sigma": np.nan,
        "n_timed": 0, "n_runs": 0, "n_wins": 0, "win_rate": 0,
        "recent_form": "", "race_gap_days": np.nan,
        "sectional_avg": np.nan, "gap_adj": 0.0, "speed_adj": 0.0,
    }

    if hist.empty:
        return result

    hist = hist.sort_values("date", ascending=False).copy()
    total = len(hist)
    wins = int((hist["finish_position"] == 1).sum())
    result["n_runs"] = total
    result["n_wins"] = wins
    result["win_rate"] = wins / total if total > 0 else 0

    # Recent form string
    recent_pos = hist.head(6)["finish_position"].tolist()
    result["recent_form"] = "".join(str(int(p)) for p in recent_pos if pd.notna(p))

    # Race gap adjustment
    gap_adj, gap_days = get_race_gap_adjustment(hist)
    result["race_gap_days"] = gap_days
    result["gap_adj"] = gap_adj

    # Early speed adjustment
    speed_adj, sect_avg = get_early_speed_adjustment(hist, race_dist, race_track)
    result["sectional_avg"] = sect_avg
    result["speed_adj"] = speed_adj

    # Find timed runs at similar distance
    has_time = hist["run_time"].notna() & (hist["run_time"] > 0)

    # Category-based matching: sprints (200-300m), middle (350-500m), stayers (500-750m+)
    def dist_category(d):
        if d <= 300:
            return "sprint"
        elif d <= 500:
            return "middle"
        else:
            return "stayer"

    race_cat = dist_category(race_dist)
    hist_cats = hist["distance"].apply(dist_category)

    # First try: exact distance match (±30m)
    exact = (hist["distance"] - race_dist).abs() <= 30
    relevant = hist[has_time & exact].copy()

    if len(relevant) == 0:
        # Second try: same category, scale proportionally (±50m)
        same_cat = hist_cats == race_cat
        close = (hist["distance"] - race_dist).abs() <= 75
        relevant = hist[has_time & same_cat & close].copy()
        if len(relevant) > 0:
            relevant = relevant.copy()
            relevant["run_time"] = relevant["run_time"] * (race_dist / relevant["distance"])

    if len(relevant) == 0:
        return result

    # Decay-weighted mean (most recent runs weighted more)
    times = relevant.head(8)["run_time"].values
    n = len(times)
    result["n_timed"] = n

    # Exponential decay weights: most recent = highest weight
    decay = 0.7
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()

    mu = np.average(times, weights=weights)

    # Apply race gap and early speed adjustments to mu
    mu += gap_adj
    mu += speed_adj

    # Estimate sigma: two components
    #   1. Intrinsic variability: how much the dog's times actually vary
    #   2. Estimation uncertainty: how unsure we are about the true mean
    default_std = get_default_std(race_dist)

    if n >= 4:
        intrinsic = np.sqrt(np.average((times - np.average(times, weights=weights)) ** 2, weights=weights))
        intrinsic = max(intrinsic, 0.12)
    elif n >= 2:
        intrinsic = max(np.std(times), 0.15)
    else:
        intrinsic = default_std

    estimation_unc = default_std / np.sqrt(n)
    sigma = np.sqrt(intrinsic ** 2 + estimation_unc ** 2)

    # Add uncertainty for cross-track runs
    if "track" in relevant.columns:
        n_diff_track = (relevant.head(n)["track"].str.lower() != race_track.lower()).sum()
        if n_diff_track > 0:
            track_penalty = 0.08 * (n_diff_track / n)
            sigma = np.sqrt(sigma ** 2 + track_penalty ** 2)

    # Add extra uncertainty for long race gaps (less predictable)
    if gap_days is not None and gap_days > 30:
        gap_unc = 0.06 * min(gap_days / 60, 1.0)
        sigma = np.sqrt(sigma ** 2 + gap_unc ** 2)

    result["est_mu"] = round(mu, 3)
    result["est_sigma"] = round(sigma, 3)
    return result


def simulate_race(mus, sigmas, traps, n_sims=N_SIMS):
    """Monte Carlo simulate a race and return win probabilities.

    Each dog's time is drawn from N(mu + trap_bias, sigma).
    The dog with the lowest time wins.
    Dogs with no time estimate get the race average + penalty with high variance.
    """
    n_dogs = len(mus)
    valid = ~np.isnan(mus)

    # For dogs with no time data, use the average of others + a penalty
    if valid.sum() > 0:
        avg_mu = np.nanmean(mus)
        avg_sigma = np.nanmax(sigmas[valid])  # use the widest sigma
    else:
        # No data for anyone — equal probabilities
        return np.ones(n_dogs) / n_dogs

    filled_mus = mus.copy()
    filled_sigmas = sigmas.copy()
    for i in range(n_dogs):
        if np.isnan(filled_mus[i]):
            filled_mus[i] = avg_mu + 0.15  # slight penalty for unknown
            filled_sigmas[i] = avg_sigma * 1.3  # more uncertainty
        # Add trap bias
        trap = int(traps[i])
        filled_mus[i] += TRAP_BIAS_SECS.get(trap, 0)

    # Simulate: each row is a simulation, each column is a dog
    times = np.random.normal(
        loc=filled_mus.reshape(1, -1),
        scale=filled_sigmas.reshape(1, -1),
        size=(n_sims, n_dogs)
    )

    # Winner = lowest time in each simulation
    winners = np.argmin(times, axis=1)
    win_counts = np.bincount(winners, minlength=n_dogs)
    probs = win_counts / n_sims

    return probs


# Generate predictions
print(f"\nSimulating {N_SIMS:,} races per contest...")
print()
print("=" * 125)
print("  ROMFORD - FRIDAY 13 MARCH 2026 - MONTE CARLO TIME-SIMULATION MODEL (v5+trap+gap+speed)")
print("=" * 125)

all_rows = []

for race_num in sorted(rc["race_number"].unique()):
    race = rc[rc["race_number"] == race_num].copy()
    dist = int(race["distance"].iloc[0])

    # Estimate time distributions for each dog
    features = []
    for _, row in race.iterrows():
        h = history[history["greyhound"] == row["greyhound"]]
        feat = estimate_time_distribution(row["greyhound"], h, dist, "Romford")
        features.append(feat)

    feat_df = pd.DataFrame(features, index=race.index)

    # Run Monte Carlo simulation
    mus = feat_df["est_mu"].values.astype(float)
    sigmas = feat_df["est_sigma"].values.astype(float)
    traps = race["trap"].values.astype(float)

    probs = simulate_race(mus, sigmas, traps)

    race["model_prob"] = probs
    race["model_rank"] = race["model_prob"].rank(ascending=False, method="min").astype(int)
    race["mkt_rank"] = race["mkt_prob"].rank(ascending=False, method="min").astype(int)
    race["edge"] = race["model_prob"] - race["mkt_prob"]

    for c in feat_df.columns:
        race[c] = feat_df[c]

    # Print race
    rt = race["race_time"].iloc[0]
    gr = race["grade"].iloc[0]
    rname = str(race["race_name"].iloc[0])[:50]

    race = race.sort_values("model_prob", ascending=False)

    print(f"\n  Race {race_num} | {rt} | {dist}m {gr} | {rname}")
    print("  " + "-" * 121)
    print(f"  {'Trap':>4}  {'Greyhound':<22} {'Odds':>7}  {'Mkt%':>6}  {'Model%':>7}  {'Edge':>7}  {'EstTime':>7} {'+-Std':>6} {'Timed':>5} {'Gap':>4} {'Sect':>5} {'Note'}")
    print("  " + "-" * 121)

    for i, (_, row) in enumerate(race.iterrows()):
        t = int(row["trap"])
        n = row["greyhound"]
        o = row["odds_frac"] if row["odds_frac"] else "-"
        mp = f"{row['mkt_prob']:.1%}" if pd.notna(row["mkt_prob"]) else "  -  "
        mdl = f"{row['model_prob']:.1%}"
        e = row["edge"]
        e_str = f"{e:+.1%}" if pd.notna(e) else "  -  "
        nt = int(row.get("n_timed", 0))
        et = f"{row['est_mu']:.2f}s" if pd.notna(row.get("est_mu")) else "   -   "
        ts = f"{row['est_sigma']:.2f}s" if pd.notna(row.get("est_sigma")) else "  -  "
        gd = row.get("race_gap_days", np.nan)
        gd_str = f"{int(gd)}d" if pd.notna(gd) else "  -"
        sa = row.get("sectional_avg", np.nan)
        sa_str = f"{sa:.2f}" if pd.notna(sa) else "  -"

        if i == 0:
            note = "<< SEL"
        elif i == 1:
            note = "   Danger"
        elif pd.notna(e) and e > 0.05:
            note = "   VALUE!"
        else:
            note = ""

        print(f"    T{t}  {n:<22} {o:>7}  {mp:>6}  {mdl:>7}  {e_str:>7}  {et:>7} {ts:>6} {nt:>5} {gd_str:>4} {sa_str:>5} {note}")

    for _, row in race.iterrows():
        all_rows.append(row)

df = pd.DataFrame(all_rows)

# Save CSV
cols = ["date", "track", "race_number", "race_time", "distance", "grade", "race_name",
        "trap", "greyhound", "odds_frac", "odds_dec", "mkt_prob_raw", "mkt_prob",
        "model_prob", "model_rank", "mkt_rank", "edge",
        "est_mu", "est_sigma", "n_timed", "best_time", "form", "trainer",
        "n_runs", "n_wins", "win_rate", "recent_form",
        "race_gap_days", "sectional_avg", "gap_adj", "speed_adj"]
avail = [c for c in cols if c in df.columns]
out = df[avail].sort_values(["race_number", "model_rank"])
out.to_csv(project_root / "data/romford_predictions_v5_2026-03-13.csv", index=False)
print(f"\nSaved to data/romford_predictions_v5_2026-03-13.csv")

print("\n" + "=" * 125)
print("  METHOD: Each dog's time modelled as Normal(EstTime, Std). 50,000 simulated races per contest.")
print("  EstTime = decay-weighted average of recent runs at similar distance (most recent = most weight)")
print("           + Romford trap bias (T1 -0.12s to T6 +0.05s, calibrated from 1481 runs)")
print("           + Race gap adjustment (30-59 day rest -> -0.06s bonus)")
print("           + Early speed factor (fast sectional times -> faster finish, ~0.08s per SD)")
print("  Std = observed variability + estimation uncertainty + cross-track/gap uncertainty")
print("  Dogs with no time data: assigned race average + 0.15s penalty, 30% extra uncertainty")
print("  Mkt% = bookmaker implied probability (overround removed)")
print("  Edge = Model% - Mkt%  (positive = model thinks undervalued)")
print("  DISCLAIMER: Estimates only. Gamble responsibly.")
print("=" * 125)
