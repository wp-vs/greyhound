"""Sweep Kelly cap with randomized race order to remove sequence bias.

For each simulation:
  1. Shuffle the 12 races into a random order
  2. Walk through sequentially, betting fraction of current bankroll
  3. Record final bankroll

This eliminates the effect of which races come first/last.
"""

import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

N_SIMS = 200_000
KELLY_FRACTION = 0.25
STARTING_BANKROLL = 100.0
MIN_EDGE = 0.02
MAX_BETS_PER_RACE = 2

df = pd.read_csv(project_root / "data/romford_predictions_v5_2026-03-13.csv")


def kelly_frac(p, decimal_odds):
    b = decimal_odds - 1
    q = 1 - p
    f = (b * p - q) / b
    return max(f, 0)


# Build race schedule
races = []
for rn in sorted(df["race_number"].unique()):
    race = df[df["race_number"] == rn].copy()

    candidates = []
    for _, row in race.iterrows():
        p = row["model_prob"]
        odds_dec = row.get("odds_dec", np.nan)
        if pd.isna(odds_dec) or pd.isna(p) or odds_dec <= 1:
            continue
        edge = p - (1.0 / odds_dec)
        if edge < MIN_EDGE:
            continue
        fk = kelly_frac(p, odds_dec)
        if fk <= 0:
            continue
        candidates.append({
            "greyhound": row["greyhound"],
            "odds_dec": odds_dec,
            "model_prob": p,
            "kelly_full": fk,
        })

    candidates.sort(key=lambda x: x["kelly_full"], reverse=True)
    top = candidates[:MAX_BETS_PER_RACE]

    # All runners for winner selection
    all_dogs = []
    all_probs = []
    for _, row in race.iterrows():
        all_dogs.append(row["greyhound"])
        all_probs.append(row["model_prob"])
    all_probs = np.array(all_probs)
    all_probs /= all_probs.sum()

    races.append({
        "race_number": rn,
        "bets": top,
        "all_dogs": all_dogs,
        "all_probs": all_probs,
    })

n_races = len(races)

# Pre-generate all race winners (for each race, N_SIMS outcomes)
print(f"Pre-generating {N_SIMS:,} x {n_races} race outcomes...")
race_winner_indices = np.zeros((n_races, N_SIMS), dtype=int)
for ri, race_info in enumerate(races):
    race_winner_indices[ri] = np.random.choice(
        len(race_info["all_dogs"]), size=N_SIMS, p=race_info["all_probs"]
    )

# Pre-generate shuffled race orders
print(f"Pre-generating {N_SIMS:,} shuffled race orders...")
race_orders = np.zeros((N_SIMS, n_races), dtype=int)
for sim in range(N_SIMS):
    race_orders[sim] = np.random.permutation(n_races)

# Sweep caps
caps = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

print(f"\nRunning {N_SIMS:,} simulations per cap, {len(caps)} caps, randomized race order...\n")
print(f"  {'Cap':>5}  {'Mean':>8}  {'Median':>8}  {'EV':>8}  {'P(prof)':>8}  {'P(ruin)':>8}  "
      f"{'5th%':>8}  {'25th%':>8}  {'75th%':>8}  {'95th%':>8}  {'Std':>8}  {'Sharpe':>7}")
print("  " + "-" * 110)

results = []

for cap in caps:
    finals = np.full(N_SIMS, STARTING_BANKROLL, dtype=float)

    for sim in range(N_SIMS):
        br = finals[sim]
        order = race_orders[sim]

        for race_idx in order:
            race_info = races[race_idx]
            bets = race_info["bets"]
            if not bets or br < 0.10:
                continue

            winner = race_info["all_dogs"][race_winner_indices[race_idx, sim]]

            for bet in bets:
                fk = bet["kelly_full"] * KELLY_FRACTION
                fk = min(fk, cap)
                stake = fk * br
                if stake < 0.10:
                    continue

                if bet["greyhound"] == winner:
                    br += stake * (bet["odds_dec"] - 1)
                else:
                    br -= stake

            br = max(br, 0)

        finals[sim] = br

    mean_br = finals.mean()
    median_br = np.median(finals)
    ev = mean_br - STARTING_BANKROLL
    profit_pct = (finals > STARTING_BANKROLL).mean()
    ruin_pct = (finals < 20).mean()
    p5 = np.percentile(finals, 5)
    p25 = np.percentile(finals, 25)
    p75 = np.percentile(finals, 75)
    p95 = np.percentile(finals, 95)
    std = finals.std()
    sharpe = ev / std if std > 0 else 0

    results.append({
        "cap": cap, "mean": mean_br, "median": median_br,
        "ev": ev, "profit_pct": profit_pct, "ruin_pct": ruin_pct,
        "p5": p5, "p25": p25, "p75": p75, "p95": p95, "std": std, "sharpe": sharpe,
    })

    print(f"  {cap:>4.0%}  £{mean_br:>7.2f}  £{median_br:>7.2f}  £{ev:>+7.2f}  "
          f"{profit_pct:>7.1%}  {ruin_pct:>7.1%}  "
          f"£{p5:>7.2f}  £{p25:>7.2f}  £{p75:>7.2f}  £{p95:>7.2f}  "
          f"£{std:>7.2f}  {sharpe:>6.3f}")

res_df = pd.DataFrame(results)

best_ev = res_df.loc[res_df["ev"].idxmax()]
best_median = res_df.loc[res_df["median"].idxmax()]
best_sharpe = res_df.loc[res_df["sharpe"].idxmax()]

print(f"\n{'=' * 115}")
print(f"  OPTIMAL KELLY CAP (randomized race order, {N_SIMS:,} sims each):")
print(f"{'=' * 115}")
print(f"  Best EV:             cap = {best_ev['cap']:>4.0%}   EV = £{best_ev['ev']:>+.2f}   median = £{best_ev['median']:.2f}   Sharpe = {best_ev['sharpe']:.3f}")
print(f"  Best median:         cap = {best_median['cap']:>4.0%}   EV = £{best_median['ev']:>+.2f}   median = £{best_median['median']:.2f}   Sharpe = {best_median['sharpe']:.3f}")
print(f"  Best risk-adjusted:  cap = {best_sharpe['cap']:>4.0%}   EV = £{best_sharpe['ev']:>+.2f}   median = £{best_sharpe['median']:.2f}   Sharpe = {best_sharpe['sharpe']:.3f}")

# Check if EV actually plateaus or keeps rising
print(f"\n  Does EV plateau? Last 5 caps:")
for _, r in res_df.tail(5).iterrows():
    print(f"    cap {r['cap']:>4.0%}: EV £{r['ev']:>+.2f}, median £{r['median']:.2f}")

res_df.to_csv(project_root / "data/kelly_cap_sweep_v2_2026-03-13.csv", index=False)
print(f"\n  Saved to data/kelly_cap_sweep_v2_2026-03-13.csv")
