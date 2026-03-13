"""Sweep Kelly cap from 1% to 100% to find optimal maximum bet size."""

import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
np.random.seed(42)

N_SIMS = 100_000
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


# Pre-build race schedule (same for all caps)
schedule = []
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

    all_probs = {}
    for _, row in race.iterrows():
        all_probs[row["greyhound"]] = row["model_prob"]

    schedule.append({"bets": top, "all_probs": all_probs})

# Pre-generate race winners for all sims (consistent across caps)
race_winners = []
for race_info in schedule:
    dogs = list(race_info["all_probs"].keys())
    probs = np.array([race_info["all_probs"][d] for d in dogs])
    probs = probs / probs.sum()
    winners = np.random.choice(len(dogs), size=N_SIMS, p=probs)
    winner_names = [dogs[w] for w in winners]
    race_winners.append(winner_names)

# Sweep caps
caps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]

results = []

for cap in caps:
    finals = np.full(N_SIMS, STARTING_BANKROLL)

    for race_idx, race_info in enumerate(schedule):
        bets = race_info["bets"]
        if not bets:
            continue

        for sim in range(N_SIMS):
            br = finals[sim]
            if br < 0.10:
                continue

            winner = race_winners[race_idx][sim]

            for bet in bets:
                fk = bet["kelly_full"] * KELLY_FRACTION
                fk = min(fk, cap)
                stake = fk * br
                if stake < 0.10:
                    continue

                if bet["greyhound"] == winner:
                    finals[sim] += stake * (bet["odds_dec"] - 1)
                else:
                    finals[sim] -= stake

            finals[sim] = max(finals[sim], 0)

    mean_br = finals.mean()
    median_br = np.median(finals)
    ev = mean_br - STARTING_BANKROLL
    median_pnl = median_br - STARTING_BANKROLL
    profit_pct = (finals > STARTING_BANKROLL).mean()
    ruin_pct = (finals < 20).mean()
    p5 = np.percentile(finals, 5)
    p95 = np.percentile(finals, 95)
    sharpe = ev / finals.std() if finals.std() > 0 else 0

    results.append({
        "cap": cap, "mean": mean_br, "median": median_br,
        "ev": ev, "median_pnl": median_pnl,
        "profit_pct": profit_pct, "ruin_pct": ruin_pct,
        "p5": p5, "p95": p95, "std": finals.std(), "sharpe": sharpe,
    })

    print(f"  Cap {cap:>5.0%}: Mean £{mean_br:>7.2f}  Median £{median_br:>7.2f}  "
          f"EV £{ev:>+7.2f}  P(profit) {profit_pct:>5.1%}  P(ruin) {ruin_pct:>4.1%}  "
          f"5th% £{p5:>7.2f}  95th% £{p95:>8.2f}  Sharpe {sharpe:.3f}")

# Find optimal
res_df = pd.DataFrame(results)

best_ev = res_df.loc[res_df["ev"].idxmax()]
best_median = res_df.loc[res_df["median"].idxmax()]
best_sharpe = res_df.loc[res_df["sharpe"].idxmax()]

print(f"\n{'=' * 100}")
print(f"  OPTIMAL KELLY CAP BY METRIC:")
print(f"{'=' * 100}")
print(f"  Best expected value:  cap = {best_ev['cap']:>5.0%}  (EV = £{best_ev['ev']:>+.2f}, median = £{best_median['median']:.2f})")
print(f"  Best median outcome:  cap = {best_median['cap']:>5.0%}  (EV = £{best_median['ev']:>+.2f}, median = £{best_median['median']:.2f})")
print(f"  Best risk-adjusted:   cap = {best_sharpe['cap']:>5.0%}  (Sharpe = {best_sharpe['sharpe']:.3f}, EV = £{best_sharpe['ev']:>+.2f})")

print(f"\n  Note: 'Sharpe' here = EV / StdDev of final bankroll (higher = better risk-adjusted)")
print(f"  Mean (EV) keeps rising with higher caps but variance explodes.")
print(f"  Median peaks then falls — that's where over-betting starts hurting the typical outcome.")

res_df.to_csv(project_root / "data/kelly_cap_sweep_2026-03-13.csv", index=False)
print(f"\n  Saved to data/kelly_cap_sweep_2026-03-13.csv")
