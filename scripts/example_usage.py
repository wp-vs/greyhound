#!/usr/bin/env python3
"""Example usage of the greyhound prediction system.

Demonstrates the full workflow:
1. Scraping historical data
2. Training models
3. Making predictions
4. Finding value bets
"""

from datetime import date

from greyhound.pipeline import Pipeline


def main():
    # --- Step 1: Scrape historical data ---
    # Uncomment to scrape data from GBGB:
    #
    # pipeline = Pipeline(data_dir="data")
    # pipeline.scrape_data(
    #     start_date=date(2024, 1, 1),
    #     end_date=date(2024, 12, 31),
    #     tracks=["Romford", "Crayford", "Hove", "Nottingham"],
    # )

    # --- Step 2: Train models ---
    pipeline = Pipeline(
        data_dir="data",
        model_dir="models/saved",
        val_split_date="2024-10-01",
    )

    # Run the full pipeline (loads data, engineers features, trains, evaluates)
    results = pipeline.run()

    if "error" in results:
        print(f"Error: {results['error']}")
        print("Make sure you have data in data/raw/ directory.")
        print("You can either:")
        print("  1. Run: greyhound scrape --start 2024-01-01 --end 2024-12-31")
        print("  2. Place CSV/parquet files in data/raw/")
        return

    # Print metrics
    print(f"\nTrained {results['n_models']} models")
    for key, val in results["metrics"].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")

    # --- Step 3: Save the pipeline ---
    pipeline.save()

    # --- Step 4: Inspect value bets ---
    value_bets = results["value_bets"]
    if not value_bets.empty:
        print(f"\nFound {len(value_bets)} value bets:")
        for _, row in value_bets.head(10).iterrows():
            print(
                f"  {row['greyhound']} @ {row.get('sp_decimal', 'N/A'):.2f} "
                f"(edge: {row['edge']:.1%})"
            )

    # --- Step 5: Predict on new data ---
    # To predict on a race card:
    #
    # import pandas as pd
    # card = pd.read_csv("racecard.csv")
    # pipeline.load()
    # card = pipeline.engineer_features(card)
    # predictions = pipeline.predict(card)
    # print(predictions[["greyhound", "trap", "predicted_win_prob", "predicted_rank"]])

    # --- Using Elo ratings independently ---
    from greyhound.features.elo import EloRating

    elo = pipeline.elo
    top_dogs = elo.top_rated(10)
    print("\nTop 10 Elo ratings:")
    for name, rating in top_dogs:
        print(f"  {name}: {rating:.0f}")

    # Predict a specific race
    runners = ["Dog A", "Dog B", "Dog C", "Dog D", "Dog E", "Dog F"]
    probs = elo.predict_race(runners)
    print("\nElo race prediction:")
    for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {name}: {prob:.1%}")


if __name__ == "__main__":
    main()
