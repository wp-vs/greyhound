"""Sequential Kelly betting strategy for tonight's Romford races.

Bets a fraction of REMAINING bankroll each race, not a fixed amount.
Simulates 100,000 possible evenings to estimate expected value,
median outcome, and risk of ruin.
"""

import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
np.random.seed(42)

N_SIMS = 100_000
KELLY_FRACTION = 0.25  # quarter Kelly
STARTING_BANKROLL = 100.0
MIN_EDGE = 0.02
MAX_BETS_PER_RACE = 2
MAX_KELLY_CAP = 0.10  # max 10% of current bankroll on any single bet

# Load predictions
df = pd.read_csv(project_root / "data/romford_predictions_v5_2026-03-13.csv")


def kelly_frac(p, decimal_odds):
    """Raw Kelly fraction (before applying fractional Kelly)."""
    b = decimal_odds - 1
    q = 1 - p
    f = (b * p - q) / b  # equivalent to p - q/b
    return max(f, 0)


# Build the ordered bet schedule: for each race, which dogs to bet on
schedule = []

for rn in sorted(df["race_number"].unique()):
    race = df[df["race_number"] == rn].copy()
    rt = race["race_time"].iloc[0]

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
            "race_number": rn,
            "race_time": rt,
            "greyhound": row["greyhound"],
            "trap": int(row["trap"]),
            "odds_frac": row["odds_frac"],
            "odds_dec": odds_dec,
            "model_prob": p,
            "mkt_prob": row.get("mkt_prob", np.nan),
            "edge": edge,
            "kelly_full": fk,
            "n_timed": int(row.get("n_timed", 0)),
        })

    candidates.sort(key=lambda x: x["kelly_full"], reverse=True)
    top = candidates[:MAX_BETS_PER_RACE]

    # Also store all runners' model probs for simulation
    all_probs = {}
    for _, row in race.iterrows():
        all_probs[row["greyhound"]] = row["model_prob"]

    for bet in top:
        bet["race_probs"] = all_probs
    schedule.append({"race_number": rn, "race_time": rt, "bets": top, "all_probs": all_probs})


# Print the strategy
print("=" * 115)
print("  ROMFORD - FRIDAY 13 MARCH 2026 - SEQUENTIAL KELLY STRATEGY")
print(f"  Starting bankroll: £{STARTING_BANKROLL:.0f} | Quarter Kelly | Bet fraction of REMAINING bankroll each race")
print("=" * 115)

# Walk through the evening with starting bankroll to show planned bets
bankroll = STARTING_BANKROLL

print(f"\n  {'Race':>4} {'Time':>5}  {'Dog':<22} {'Trap':>4} {'Odds':>7} {'Model%':>7} {'Mkt%':>6} {'Edge':>6}"
      f" {'Kelly%':>7} {'Bankroll':>9} {'Stake':>7} {'If wins':>9}")
print("  " + "-" * 115)

planned_bets = []
for race_info in schedule:
    rn = race_info["race_number"]
    rt = race_info["race_time"]
    bets = race_info["bets"]

    if not bets:
        print(f"  R{rn:>2} {rt:>5}  {'(no value bets)':<22}")
        continue

    race_total_stake = 0
    for bet in bets:
        fk = bet["kelly_full"] * KELLY_FRACTION
        fk = min(fk, MAX_KELLY_CAP)
        stake = round(fk * bankroll, 2)
        if stake < 0.10:  # minimum meaningful bet
            continue
        win_return = stake * bet["odds_dec"]

        conf = "***" if bet["n_timed"] >= 4 else "**" if bet["n_timed"] >= 2 else "*" if bet["n_timed"] >= 1 else "?"

        print(f"  R{rn:>2} {rt:>5}  {bet['greyhound']:<22} T{bet['trap']:>1} {bet['odds_frac']:>7}"
              f" {bet['model_prob']:>6.1%} {bet['mkt_prob']:>5.1%} {bet['edge']:>+5.1%}"
              f" {fk:>6.1%}  £{bankroll:>7.2f} £{stake:>5.2f}  £{win_return:>7.2f} {conf}")

        planned_bets.append({
            "race_number": rn, "race_time": rt,
            "greyhound": bet["greyhound"], "trap": bet["trap"],
            "odds_frac": bet["odds_frac"], "odds_dec": bet["odds_dec"],
            "model_prob": bet["model_prob"], "mkt_prob": bet["mkt_prob"],
            "edge": bet["edge"], "kelly_pct": fk,
            "bankroll_before": bankroll, "stake": stake,
            "win_return": win_return, "confidence": conf,
        })
        race_total_stake += stake

    # Deduct stakes from bankroll (we don't know outcomes yet)
    bankroll -= race_total_stake

print("  " + "-" * 115)
total_staked = sum(b["stake"] for b in planned_bets)
print(f"\n  Total planned stakes: £{total_staked:.2f}")
print(f"  Remaining after all bets placed: £{bankroll:.2f}")


# Now simulate 100,000 possible evenings
print(f"\n\n{'=' * 115}")
print(f"  MONTE CARLO SIMULATION: {N_SIMS:,} possible evenings")
print(f"{'=' * 115}")

final_bankrolls = np.zeros(N_SIMS)

for sim in range(N_SIMS):
    br = STARTING_BANKROLL

    for race_info in schedule:
        bets = race_info["bets"]
        all_probs = race_info["all_probs"]

        if not bets:
            continue

        # Determine winner of this race (sample from model probabilities)
        dogs = list(all_probs.keys())
        probs = np.array([all_probs[d] for d in dogs])
        probs = probs / probs.sum()  # normalize
        winner = dogs[np.random.choice(len(dogs), p=probs)]

        # Place bets as fraction of current bankroll
        for bet in bets:
            fk = bet["kelly_full"] * KELLY_FRACTION
            fk = min(fk, MAX_KELLY_CAP)
            stake = fk * br
            if stake < 0.10:
                continue

            if bet["greyhound"] == winner:
                # Win: get stake back + profit
                br += stake * (bet["odds_dec"] - 1)
            else:
                # Lose: lose stake
                br -= stake

        # Don't go below zero
        br = max(br, 0)

    final_bankrolls[sim] = br

# Analysis
print(f"\n  Starting bankroll: £{STARTING_BANKROLL:.0f}")
print(f"  Mean final bankroll: £{final_bankrolls.mean():.2f}")
print(f"  Median final bankroll: £{np.median(final_bankrolls):.2f}")
print(f"  Std deviation: £{final_bankrolls.std():.2f}")

percentiles = [5, 10, 25, 50, 75, 90, 95]
print(f"\n  Percentile outcomes:")
for p in percentiles:
    val = np.percentile(final_bankrolls, p)
    pnl = val - STARTING_BANKROLL
    print(f"    {p:>3}th percentile: £{val:>8.2f}  (P&L: £{pnl:>+8.2f})")

profit_pct = (final_bankrolls > STARTING_BANKROLL).mean()
loss_pct = (final_bankrolls < STARTING_BANKROLL).mean()
ruin_pct = (final_bankrolls < 20).mean()  # "ruin" = less than £20 left
double_pct = (final_bankrolls >= 200).mean()

print(f"\n  Probability of profit: {profit_pct:.1%}")
print(f"  Probability of loss: {loss_pct:.1%}")
print(f"  Probability of ruin (<£20): {ruin_pct:.1%}")
print(f"  Probability of doubling up (£200+): {double_pct:.1%}")

ev = final_bankrolls.mean() - STARTING_BANKROLL
print(f"\n  Expected P&L: £{ev:+.2f} ({ev/STARTING_BANKROLL:+.1%} return)")
print(f"  Median P&L: £{np.median(final_bankrolls) - STARTING_BANKROLL:+.2f}")

# Distribution histogram (text-based)
print(f"\n  Final bankroll distribution:")
bins = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 500, 1000, float("inf")]
labels = ["£0-20", "£20-40", "£40-60", "£60-80", "£80-100",
          "£100-120", "£120-150", "£150-200", "£200-300", "£300-500", "£500-1000", "£1000+"]
for i in range(len(bins) - 1):
    count = ((final_bankrolls >= bins[i]) & (final_bankrolls < bins[i + 1])).sum()
    pct = count / N_SIMS
    bar = "#" * int(pct * 100)
    print(f"    {labels[i]:>10}: {pct:>5.1%} {bar}")

# Save planned bets
bets_df = pd.DataFrame(planned_bets)
bets_df["Odds"] = "'" + bets_df["odds_frac"].astype(str)
out_cols = ["race_number", "race_time", "greyhound", "trap", "Odds",
            "model_prob", "edge", "kelly_pct", "bankroll_before", "stake",
            "win_return", "confidence"]
bets_df[out_cols].to_csv(project_root / "data/romford_sequential_bets_2026-03-13.csv", index=False)
print(f"\n  Saved bet schedule to data/romford_sequential_bets_2026-03-13.csv")

print("\n" + "=" * 115)
print("  NOTE: Expected value assumes model probabilities are correct.")
print("  If the model is miscalibrated (likely), actual results will differ.")
print("  The median is more informative than the mean for skewed distributions.")
print("  DISCLAIMER: Estimates only. Gamble responsibly.")
print("=" * 115)
