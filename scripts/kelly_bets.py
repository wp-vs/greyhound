"""Kelly criterion optimized bets for tonight's Romford races.

Kelly fraction: f* = (b*p - q) / b
  where b = decimal odds - 1, p = model win prob, q = 1 - p

Only bets where f* > 0 (positive edge) are considered.
Uses fractional Kelly (25%) to reduce variance.
Max 2 bets per race.
"""

import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Load predictions
df = pd.read_csv(project_root / "data/romford_predictions_v5_2026-03-13.csv")

KELLY_FRACTION = 0.25  # quarter Kelly — full Kelly is too aggressive
BANKROLL = 100.0  # £100 starting bankroll
MIN_EDGE = 0.02  # minimum 2% edge to bet
MAX_BETS_PER_RACE = 2
MAX_KELLY_CAP = 0.10  # cap any single bet at 10% of bankroll


def kelly_fraction(p, decimal_odds, fraction=KELLY_FRACTION):
    """Compute fractional Kelly stake as fraction of bankroll."""
    b = decimal_odds - 1  # net odds (profit per £1 staked)
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        return 0.0
    return min(f_star * fraction, MAX_KELLY_CAP)


print("=" * 110)
print(f"  ROMFORD - FRIDAY 13 MARCH 2026 - KELLY CRITERION BETS")
print(f"  Bankroll: £{BANKROLL:.0f} | Kelly fraction: {KELLY_FRACTION:.0%} | Min edge: {MIN_EDGE:.0%} | Max 2 bets/race")
print("=" * 110)

all_bets = []

for rn in sorted(df["race_number"].unique()):
    race = df[df["race_number"] == rn].copy()
    rt = race["race_time"].iloc[0]
    dist = int(race["distance"].iloc[0])
    gr = race["grade"].iloc[0]

    # Find bets with positive edge
    candidates = []
    for _, row in race.iterrows():
        p = row["model_prob"]
        odds_dec = row.get("odds_dec", np.nan)
        if pd.isna(odds_dec) or pd.isna(p) or odds_dec <= 1:
            continue

        edge = p - (1.0 / odds_dec)
        if edge < MIN_EDGE:
            continue

        f = kelly_fraction(p, odds_dec)
        if f <= 0:
            continue

        candidates.append({
            "race_number": rn,
            "race_time": rt,
            "distance": dist,
            "grade": gr,
            "greyhound": row["greyhound"],
            "trap": int(row["trap"]),
            "odds_frac": row["odds_frac"],
            "odds_dec": odds_dec,
            "mkt_prob": row["mkt_prob"],
            "model_prob": p,
            "edge": edge,
            "kelly_full": (((odds_dec - 1) * p) - (1 - p)) / (odds_dec - 1),
            "kelly_frac": f,
            "stake": round(f * BANKROLL, 2),
            "est_time": row.get("est_mu", np.nan),
            "n_timed": int(row.get("n_timed", 0)),
        })

    # Sort by Kelly fraction (best bet first), take top 2
    candidates.sort(key=lambda x: x["kelly_frac"], reverse=True)
    top = candidates[:MAX_BETS_PER_RACE]

    print(f"\n  Race {rn} | {rt} | {dist}m {gr}")
    print("  " + "-" * 106)

    if not top:
        print("    No value bets found (no positive edge exceeding minimum threshold)")
    else:
        print(f"  {'Trap':>4}  {'Greyhound':<22} {'Odds':>7} {'Mkt%':>6} {'Model%':>7} {'Edge':>6} {'Kelly%':>7} {'Stake':>7} {'Pot.Ret':>8} {'Timed':>5}")
        print("  " + "-" * 106)
        for b in top:
            pot_return = b["stake"] * b["odds_dec"]
            confidence = "***" if b["n_timed"] >= 4 else "**" if b["n_timed"] >= 2 else "*" if b["n_timed"] >= 1 else "?"
            print(f"    T{b['trap']}  {b['greyhound']:<22} {b['odds_frac']:>7} {b['mkt_prob']:>5.1%} {b['model_prob']:>6.1%} {b['edge']:>+5.1%} {b['kelly_frac']:>6.1%}  £{b['stake']:>5.2f}  £{pot_return:>6.2f} {b['n_timed']:>4} {confidence}")
            all_bets.append(b)

# Summary
print("\n" + "=" * 110)
print("  BET SUMMARY")
print("=" * 110)

if all_bets:
    bets_df = pd.DataFrame(all_bets)
    total_staked = bets_df["stake"].sum()
    total_pot_return = (bets_df["stake"] * bets_df["odds_dec"]).sum()

    # Expected value calculation
    ev = sum(b["stake"] * (b["model_prob"] * (b["odds_dec"] - 1) - (1 - b["model_prob"])) for b in all_bets)

    print(f"\n  Total bets: {len(all_bets)} across {bets_df['race_number'].nunique()} races")
    print(f"  Total staked: £{total_staked:.2f} of £{BANKROLL:.0f} bankroll ({total_staked/BANKROLL:.1%})")
    print(f"  Expected value: £{ev:+.2f} (EV% on stakes: {ev/total_staked:+.1%})" if total_staked > 0 else "")
    print(f"\n  {'Race':>4} {'Time':>5}  {'Trap':>4}  {'Greyhound':<22} {'Odds':>7}  {'Stake':>7}  {'Win Ret':>8}  {'Model%':>7}  {'Edge':>6}")
    print("  " + "-" * 90)
    for b in sorted(all_bets, key=lambda x: x["race_number"]):
        win_ret = b["stake"] * b["odds_dec"]
        print(f"  R{b['race_number']:>2} {b['race_time']:>5}   T{b['trap']}  {b['greyhound']:<22} {b['odds_frac']:>7}  £{b['stake']:>5.2f}  £{win_ret:>6.2f}  {b['model_prob']:>6.1%}  {b['edge']:>+5.1%}")

    print("  " + "-" * 90)
    print(f"  {'':>4} {'':>5}  {'':>4}  {'TOTAL':<22} {'':>7}  £{total_staked:>5.2f}")

    # Confidence breakdown
    print(f"\n  Confidence key: *** = 4+ timed runs at distance, ** = 2-3, * = 1, ? = no time data")

    # Best bets highlight
    print(f"\n  TOP 3 BETS (by Kelly fraction):")
    top3 = sorted(all_bets, key=lambda x: x["kelly_frac"], reverse=True)[:3]
    for i, b in enumerate(top3, 1):
        print(f"    {i}. R{b['race_number']} T{b['trap']} {b['greyhound']} @ {b['odds_frac']} — £{b['stake']:.2f} (edge {b['edge']:+.1%}, model {b['model_prob']:.1%})")

    # Save bets
    bets_df.to_csv(project_root / "data/romford_kelly_bets_2026-03-13.csv", index=False)
    print(f"\n  Saved to data/romford_kelly_bets_2026-03-13.csv")

print("\n" + "=" * 110)
print("  DISCLAIMER: These are model estimates only. Greyhound racing involves significant")
print("  randomness (trap breaks, bumping, going conditions) not captured by time-based models.")
print("  Quarter Kelly is used to reduce risk. Never bet more than you can afford to lose.")
print("=" * 110)
