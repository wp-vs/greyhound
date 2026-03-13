"""Full Kelly strategy with estimated times, comparing outright vs each-way.

E/W terms for greyhounds: 1/4 odds, top 2 places in 6-runner races.
An E/W bet is two equal bets: one to win, one to place.
"""

import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
np.random.seed(42)

N_SIMS = 100_000
KELLY_FRACTION = 1.0
BET_CAP = 0.50
STARTING_BANKROLL = 100.0
MIN_EDGE = 0.02
MAX_BETS_PER_RACE = 2
EW_FRACTION = 1/4
EW_PLACES = 2

# Updated odds
odds_data = {
    1: {"Bonville Secret": "11/4", "Goulane Vegas": "11/4", "Droopys Rhona": "7/2",
        "Abel Mabel": "9/2", "Bluejig Designer": "9/2", "Olwinn Seen It": "7"},
    2: {"Calypso Blaze": "5/4", "Lil Bo Beep": "11/4", "Headford Wayne": "11/2",
        "Aero Espresso": "7", "Skidroe Lady": "8", "Hellofalooker": "16"},
    3: {"Rockmount Kellie": "7/4", "Mahoonagh Hoffa": "10/3", "Bretons Girl": "9/2",
        "My Lil Bella": "7", "Brindle Moon": "10", "Jayar Rogue": "10"},
    4: {"Bretons Boy": "3", "Lesleys Buddy": "3", "Stradeen Spirit": "7/2",
        "Alan The First": "9/2", "Piemans Fletch": "9/2", "Bombay Trend": "8"},
    5: {"Aayamza Legend": "2", "Farneys Tearaway": "7/2", "Droopys Rosie": "4",
        "Ballymac Florie": "9/2", "Miami Yeats": "6", "Droopys Maybe": "7"},
    6: {"Slingshot Poppy": "1/2", "Flashing Fender": "10/3", "Da Don": "6",
        "Yahoo Mareike": "14", "Droopys Will": "20", "Broadway Steel": "33"},
    7: {"Bockos Buster": "11/18", "Jacktavern Don": "11/2", "Sehnsa Amigo": "13/2",
        "Quinton Boy": "8", "Uncle Ed": "11", "Charlie Loves Me": "18"},
    8: {"Piemans Goalie": "7/4", "Swift Hostile": "28/15", "Pro Parker": "11/2",
        "Bombay Buck": "7", "Funky Adz": "8", "Tally Ho Socks": "20"},
    9: {"Unmistakeable": "1", "Alana The Second": "5/2", "Cooladerrydancer": "13/2",
        "Essjay Sonia": "13/2", "Letter Cutie": "14", "Makeit Poppy": "14"},
    10: {"Underground Jim": "6/4", "Aero Leg It": "3", "Thorpys Legacy": "9/2",
         "Ivanhoe Rose": "13/2", "Ballymac Suntan": "7", "Real Gone Lover": "10"},
    11: {"Out The Blue": "2", "Royal Hotshot": "2", "Bacon Frazzles": "9/2",
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


# Load predictions and update odds
preds = pd.read_csv(project_root / "data/romford_predictions_v5_2026-03-13.csv")

for race_num, odds in odds_data.items():
    race_mask = preds["race_number"] == race_num
    total_imp = 0
    for dog, frac in odds.items():
        dec = parse_odds(frac)
        imp = 1.0 / dec
        total_imp += imp
        mask = race_mask & (preds["greyhound"] == dog.strip().title())
        preds.loc[mask, "odds_frac"] = frac
        preds.loc[mask, "odds_dec"] = round(dec, 2)
        preds.loc[mask, "mkt_prob_raw"] = round(imp, 4)
    if total_imp > 0:
        for dog, frac in odds.items():
            mask = race_mask & (preds["greyhound"] == dog.strip().title())
            raw = preds.loc[mask, "mkt_prob_raw"].values
            if len(raw) > 0:
                preds.loc[mask, "mkt_prob"] = round(raw[0] / total_imp, 4)

# Load history for time estimates
history = pd.read_csv(project_root / "data/romford_combined_history_2026-03-13.csv")
history["greyhound"] = history["greyhound"].str.strip().str.title()
history = history[history["date"] < "2026-03-13"]

DEFAULT_STD = {225: 0.25, 400: 0.35, 575: 0.45, 750: 0.55}
TRAP_BIAS = {1: -0.12, 2: 0.02, 3: -0.06, 4: -0.02, 5: 0.03, 6: 0.05}


def get_default_std(dist):
    dists = sorted(DEFAULT_STD.keys())
    if dist <= dists[0]:
        return DEFAULT_STD[dists[0]]
    if dist >= dists[-1]:
        return DEFAULT_STD[dists[-1]]
    for i in range(len(dists) - 1):
        if dists[i] <= dist <= dists[i + 1]:
            f = (dist - dists[i]) / (dists[i + 1] - dists[i])
            return DEFAULT_STD[dists[i]] * (1 - f) + DEFAULT_STD[dists[i + 1]] * f
    return 0.35


def dist_category(d):
    if d <= 300:
        return "sprint"
    elif d <= 500:
        return "middle"
    return "stayer"


def estimate_time(name, hist, race_dist):
    has_time = hist["run_time"].notna() & (hist["run_time"] > 0)
    hist = hist.sort_values("date", ascending=False)
    exact = (hist["distance"] - race_dist).abs() <= 30
    relevant = hist[has_time & exact]
    if len(relevant) == 0:
        race_cat = dist_category(race_dist)
        hist_cats = hist["distance"].apply(dist_category)
        same_cat = hist_cats == race_cat
        close = (hist["distance"] - race_dist).abs() <= 75
        relevant = hist[has_time & same_cat & close].copy()
        if len(relevant) > 0:
            relevant = relevant.copy()
            relevant["run_time"] = relevant["run_time"] * (race_dist / relevant["distance"])
    if len(relevant) == 0:
        return np.nan, np.nan, 0
    times = relevant.head(8)["run_time"].values
    n = len(times)
    decay = 0.7
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()
    mu = np.average(times, weights=weights)
    default_std = get_default_std(race_dist)
    if n >= 4:
        intrinsic = max(np.sqrt(np.average((times - mu) ** 2, weights=weights)), 0.12)
    elif n >= 2:
        intrinsic = max(np.std(times), 0.15)
    else:
        intrinsic = default_std
    estimation_unc = default_std / np.sqrt(n)
    sigma = np.sqrt(intrinsic ** 2 + estimation_unc ** 2)
    return mu, sigma, n


# Monte Carlo: get win AND place probabilities
print("Simulating win and place probabilities...")

race_info = {}
for rn in sorted(preds["race_number"].unique()):
    race = preds[preds["race_number"] == rn].copy()
    dist = int(race["distance"].iloc[0])
    dogs = race["greyhound"].tolist()
    traps = race["trap"].tolist()

    mus, sigmas, n_timeds = [], [], []
    for _, row in race.iterrows():
        h = history[history["greyhound"] == row["greyhound"]]
        mu, sigma, nt = estimate_time(row["greyhound"], h, dist)
        mus.append(mu)
        sigmas.append(sigma)
        n_timeds.append(nt)

    mus = np.array(mus, dtype=float)
    sigmas = np.array(sigmas, dtype=float)
    valid = ~np.isnan(mus)

    avg_mu = np.nanmean(mus) if valid.sum() > 0 else 25.0
    avg_sigma = np.nanmax(sigmas[valid]) if valid.sum() > 0 else 0.5

    filled_mus = mus.copy()
    filled_sigmas = sigmas.copy()
    for i in range(len(dogs)):
        if np.isnan(filled_mus[i]):
            filled_mus[i] = avg_mu + 0.15
            filled_sigmas[i] = avg_sigma * 1.3
        filled_mus[i] += TRAP_BIAS.get(int(traps[i]), 0)

    sim_times = np.random.normal(
        loc=filled_mus.reshape(1, -1),
        scale=filled_sigmas.reshape(1, -1),
        size=(N_SIMS, len(dogs))
    )

    winners = np.argmin(sim_times, axis=1)
    win_counts = np.bincount(winners, minlength=len(dogs))
    win_probs = win_counts / N_SIMS

    rankings = np.argsort(sim_times, axis=1)
    place_counts = np.zeros(len(dogs))
    for p in range(EW_PLACES):
        for sim in range(N_SIMS):
            place_counts[rankings[sim, p]] += 1
    place_probs = place_counts / N_SIMS

    race_info[rn] = {}
    for i, dog in enumerate(dogs):
        race_info[rn][dog] = {
            "win_prob": win_probs[i],
            "place_prob": place_probs[i],
            "est_time": mus[i],
            "est_sigma": sigmas[i],
            "n_timed": n_timeds[i],
        }


def kelly_outright(p_win, odds_dec):
    b = odds_dec - 1
    q = 1 - p_win
    f = (b * p_win - q) / b
    return max(f, 0)


def kelly_ew(p_win, p_place, odds_dec):
    """Numerical Kelly for each-way via grid search."""
    place_odds_dec = 1 + (odds_dec - 1) * EW_FRACTION
    p_place_only = p_place - p_win
    p_lose = 1 - p_place

    best_f = 0
    best_growth = 0
    for f in np.arange(0.001, 0.95, 0.001):
        # Returns per GBP1 total E/W stake for each outcome:
        ret_win = 0.5 * odds_dec + 0.5 * place_odds_dec  # both halves pay
        ret_place = 0.5 * place_odds_dec                  # only place half pays
        ret_lose = 0                                       # lose everything

        g = (p_win * np.log(1 + f * (ret_win - 1))
             + p_place_only * np.log(1 + f * (ret_place - 1))
             + p_lose * np.log(1 - f))

        if g > best_growth:
            best_growth = g
            best_f = f
    return best_f


# Build strategy
print()
print("=" * 140)
print("  ROMFORD - FRIDAY 13 MARCH 2026 - FULL KELLY (50% cap) - TIMES & E/W ANALYSIS")
print("  Starting bankroll: GBP 100 | E/W terms: 1/4 odds, top 2 places")
print("=" * 140)

print(f"\n  {'Race':>4} {'Time':>5}  {'Dog':<20} {'T':>1} {'Odds':>6} {'EstT':>7}"
      f" {'Win%':>5} {'Plc%':>5} {'Edge':>6}"
      f" {'O/R Kly':>7} {'E/W Kly':>7} {'Best':>4}"
      f" {'Bankrl':>7} {'Stake':>6} {'If wins':>8} {'Conf':>4}")
print("  " + "-" * 136)

bankroll = STARTING_BANKROLL
planned_bets = []

for rn in sorted(preds["race_number"].unique()):
    race = preds[preds["race_number"] == rn].copy()
    rt = race["race_time"].iloc[0]

    candidates = []
    for _, row in race.iterrows():
        dog = row["greyhound"]
        odds_dec = row.get("odds_dec", np.nan)
        if pd.isna(odds_dec) or odds_dec <= 1:
            continue
        rd = race_info[rn][dog]
        p_win = rd["win_prob"]
        p_place = rd["place_prob"]
        edge = p_win - (1.0 / odds_dec)
        if edge < MIN_EDGE:
            continue

        k_or = kelly_outright(p_win, odds_dec)
        k_ew = kelly_ew(p_win, p_place, odds_dec)

        if k_or <= 0 and k_ew <= 0:
            continue

        best = "E/W" if k_ew > k_or else "O/R"
        best_k = max(k_or, k_ew)

        # Market implied probabilities
        mkt_win = row.get("mkt_prob", np.nan)
        if pd.isna(mkt_win):
            mkt_win = 1.0 / odds_dec
        # Market place prob: approximate using Harville model
        # P(place) ≈ 1 - (1 - P(win))^EW_PLACES for small fields
        # For 6-runner, top-2: a reasonable approximation
        mkt_place = 1 - (1 - mkt_win) ** EW_PLACES

        candidates.append({
            "race_number": rn, "race_time": rt, "greyhound": dog,
            "trap": int(row["trap"]), "odds_frac": row["odds_frac"],
            "odds_dec": odds_dec, "est_time": rd["est_time"],
            "est_sigma": rd["est_sigma"],
            "win_prob": p_win, "place_prob": p_place,
            "mkt_win_prob": mkt_win, "mkt_place_prob": mkt_place,
            "edge": edge, "kelly_or": k_or, "kelly_ew": k_ew,
            "best_type": best, "best_kelly": best_k,
            "n_timed": rd["n_timed"],
        })

    candidates.sort(key=lambda x: x["best_kelly"], reverse=True)
    top = candidates[:MAX_BETS_PER_RACE]

    if not top:
        print(f"  R{rn:>2} {rt:>5}  {'(no value bets)':<20}")
        continue

    race_total = 0
    for bet in top:
        fk = min(bet["best_kelly"], BET_CAP)
        stake = round(fk * bankroll, 2)
        if stake < 0.10:
            continue

        if bet["best_type"] == "E/W":
            place_odds = 1 + (bet["odds_dec"] - 1) * EW_FRACTION
            win_return = stake * (0.5 * bet["odds_dec"] + 0.5 * place_odds)
        else:
            win_return = stake * bet["odds_dec"]

        conf = "***" if bet["n_timed"] >= 4 else "**" if bet["n_timed"] >= 2 else "*" if bet["n_timed"] >= 1 else "?"
        et = f"{bet['est_time']:.2f}s" if not np.isnan(bet["est_time"]) else "   -   "

        print(f"  R{rn:>2} {rt:>5}  {bet['greyhound']:<20} T{bet['trap']} {bet['odds_frac']:>6} {et:>7}"
              f" {bet['win_prob']:>4.1%} {bet['place_prob']:>4.1%} {bet['edge']:>+5.1%}"
              f" {bet['kelly_or']:>6.1%} {bet['kelly_ew']:>6.1%} {bet['best_type']:>4}"
              f" {bankroll:>6.1f} {stake:>5.2f}  {win_return:>7.2f} {conf}")

        planned_bets.append({
            "Race": f"R{rn}", "Time": rt, "Dog": bet["greyhound"],
            "Trap": f"T{bet['trap']}", "Odds": "'" + str(bet["odds_frac"]),
            "EstTime": et,
            "ModelWin%": f"{bet['win_prob']:.1%}",
            "ModelPlace%": f"{bet['place_prob']:.1%}",
            "MktWin%": f"{bet['mkt_win_prob']:.1%}",
            "MktPlace%": f"{bet['mkt_place_prob']:.1%}",
            "Edge": f"{bet['edge']:+.1%}",
            "Type": bet["best_type"],
            "KellyPct": f"{fk:.1%}", "Stake": f"{stake:.2f}",
            "Confidence": conf,
        })
        race_total += stake

    bankroll -= race_total

print("  " + "-" * 136)
total = sum(float(b["Stake"]) for b in planned_bets)
print(f"\n  Total staked: GBP {total:.2f} | Held back: GBP {bankroll:.2f}")

n_or = sum(1 for b in planned_bets if b["Type"] == "O/R")
n_ew = sum(1 for b in planned_bets if b["Type"] == "E/W")
print(f"  Bet types: {n_or} outright, {n_ew} each-way")

out_df = pd.DataFrame(planned_bets)
out_path = project_root / "data/romford_fullkelly_ew_v2_2026-03-13.csv"
out_df.to_csv(out_path, index=False)
print(f"\n  Saved to {out_path.name}")

print("\n" + "=" * 140)
print("  O/R Kelly = optimal fraction for outright win bet")
print("  E/W Kelly = optimal fraction for each-way bet (1/4 odds, top 2 places)")
print("  E/W is better when a dog has high place probability relative to win probability")
print("  (e.g. consistent top-2 finisher at longer odds — the place insurance adds value)")
print("=" * 140)
