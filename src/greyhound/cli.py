"""Command-line interface for the greyhound prediction system."""

import logging
import sys
from datetime import date, datetime

import click
from rich.console import Console
from rich.table import Table

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool):
    """Greyhound race prediction system."""
    setup_logging(verbose)


@cli.command()
@click.option("--start", required=True, type=str, help="Start date (YYYY-MM-DD).")
@click.option("--end", required=True, type=str, help="End date (YYYY-MM-DD).")
@click.option("--tracks", type=str, default=None,
              help="Comma-separated track names. Default: all tracks.")
@click.option("--output-dir", type=str, default="data/raw", help="Output directory.")
def scrape(start: str, end: str, tracks: str | None, output_dir: str):
    """Scrape historical race results from GBGB."""
    from .data.scraper import GreyhoundScraper

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    track_list = [t.strip() for t in tracks.split(",")] if tracks else None

    console.print(f"Scraping results from {start_date} to {end_date}...")
    if track_list:
        console.print(f"Tracks: {', '.join(track_list)}")

    scraper = GreyhoundScraper(output_dir=output_dir)
    races = scraper.scrape_results(start_date, end_date, track_list)

    if races:
        filename = f"results_{start}_{end}.parquet"
        path = scraper.save_results(races, filename)
        console.print(f"[green]Saved {len(races)} races to {path}[/green]")
    else:
        console.print("[yellow]No races found for the specified period.[/yellow]")


@cli.command()
@click.option("--data-dir", type=str, default="data", help="Data directory.")
@click.option("--model-dir", type=str, default="models/saved", help="Model output directory.")
@click.option("--val-split", type=str, default=None,
              help="Validation split date (YYYY-MM-DD). Default: last 20%.")
def train(data_dir: str, model_dir: str, val_split: str | None):
    """Train prediction models on historical data."""
    from .pipeline import Pipeline

    pipeline = Pipeline(data_dir=data_dir, model_dir=model_dir, val_split_date=val_split)

    console.print("Loading data...")
    df = pipeline.load_data()
    if df.empty:
        console.print("[red]No data found. Run 'greyhound scrape' first.[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(df)} results, {df['greyhound'].nunique()} greyhounds")

    results = pipeline.run(df)

    # Display metrics
    metrics = results["metrics"]
    table = Table(title="Model Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, val in metrics.items():
        if isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))

    console.print(table)

    # Save pipeline
    pipeline.save()
    console.print(f"[green]Pipeline saved to {model_dir}/pipeline.joblib[/green]")


@cli.command()
@click.option("--model-dir", type=str, default="models/saved", help="Model directory.")
@click.option("--data-file", type=str, required=True,
              help="Path to race card data (CSV or parquet).")
@click.option("--min-edge", type=float, default=0.05,
              help="Minimum probability edge for value selections.")
def predict(model_dir: str, data_file: str, min_edge: float):
    """Generate predictions for upcoming races."""
    import pandas as pd

    from .pipeline import Pipeline

    pipeline = Pipeline(model_dir=model_dir)
    pipeline.load()

    # Load race card
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_file)
    else:
        df = pd.read_csv(data_file)

    console.print(f"Loaded {len(df)} runners from {data_file}")

    # Engineer features and predict
    df = pipeline.engineer_features(df)
    df = pipeline.predict(df)

    # Display predictions by race
    for race_id, race in df.groupby("race_id"):
        race = race.sort_values("predicted_win_prob", ascending=False)

        table = Table(title=f"Race: {race_id}")
        table.add_column("Trap", style="cyan", width=4)
        table.add_column("Greyhound", style="white")
        table.add_column("Win Prob", style="green", width=8)
        table.add_column("SP", style="yellow", width=6)
        table.add_column("Edge", style="magenta", width=8)

        for _, row in race.iterrows():
            edge = ""
            if "sp_implied_prob" in df.columns and not pd.isna(row.get("sp_implied_prob")):
                edge_val = row["predicted_win_prob"] - row["sp_implied_prob"]
                edge = f"{edge_val:+.1%}"

            sp = f"{row['sp_decimal']:.2f}" if "sp_decimal" in df.columns and not pd.isna(
                row.get("sp_decimal")
            ) else "-"

            table.add_row(
                str(int(row.get("trap", 0))),
                str(row.get("greyhound", "")),
                f"{row['predicted_win_prob']:.1%}",
                sp,
                edge,
            )

        console.print(table)
        console.print()

    # Show value selections
    pipeline.value_detector.min_edge = min_edge
    value = pipeline.find_value(df)
    if not value.empty:
        console.print(f"\n[bold green]Value Selections (edge >= {min_edge:.0%}):[/bold green]")
        vtable = Table()
        vtable.add_column("Race")
        vtable.add_column("Greyhound")
        vtable.add_column("Model Prob")
        vtable.add_column("Market Prob")
        vtable.add_column("Edge")
        vtable.add_column("SP")

        for _, row in value.iterrows():
            vtable.add_row(
                str(row.get("race_id", "")),
                str(row.get("greyhound", "")),
                f"{row['predicted_win_prob']:.1%}",
                f"{row.get('market_prob', 0):.1%}",
                f"{row.get('edge', 0):+.1%}",
                f"{row.get('sp_decimal', 0):.2f}",
            )

        console.print(vtable)


@cli.command("fetch-bsp")
@click.option("--start", required=True, type=str, help="Start date (YYYY-MM-DD).")
@click.option("--end", required=True, type=str, help="End date (YYYY-MM-DD).")
@click.option("--market", type=click.Choice(["win", "place"]), default="win",
              help="Market type to download.")
@click.option("--output-dir", type=str, default="data/raw/betfair_bsp",
              help="Output directory for BSP files.")
@click.option("--save-parquet/--no-parquet", default=True,
              help="Save combined parquet after download.")
def fetch_bsp(start: str, end: str, market: str, output_dir: str, save_parquet: bool):
    """Download Betfair BSP historical data for greyhounds."""
    from .data.scraper import BetfairBSPLoader

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    total_days = (end_date - start_date).days + 1

    console.print(f"Downloading Betfair BSP ({market}) data: {start_date} to {end_date} ({total_days} days)")

    loader = BetfairBSPLoader(output_dir=output_dir)
    files = loader.download_range(start_date, end_date, market=market)

    console.print(f"[green]Downloaded {len(files)} files to {output_dir}[/green]")

    if save_parquet and files:
        df = loader.load_all()
        if not df.empty:
            filename = f"bsp_{market}_{start}_{end}.parquet"
            path = loader.save_parquet(df, filename)
            console.print(f"[green]Saved {len(df):,} records to {path}[/green]")

            # Show summary
            console.print(f"\n[bold]BSP Data Summary[/bold]")
            console.print(f"  Records:     {len(df):,}")
            console.print(f"  Date range:  {df['date'].min()} to {df['date'].max()}")
            if "track" in df.columns:
                console.print(f"  Tracks:      {df['track'].nunique()}")
            if "greyhound" in df.columns:
                console.print(f"  Greyhounds:  {df['greyhound'].nunique():,}")
            if "bsp_win" in df.columns:
                winners = (df["bsp_win"] == 1).sum()
                console.print(f"  Winners:     {winners:,} ({winners / len(df):.1%})")
        else:
            console.print("[yellow]No valid BSP data found in downloaded files.[/yellow]")


@cli.command()
@click.option("--data-dir", type=str, default="data", help="Data directory.")
def stats(data_dir: str):
    """Show dataset statistics."""
    from .data.scraper import HistoricalDataLoader

    loader = HistoricalDataLoader(data_dir)
    df = loader.load_all()

    if df.empty:
        console.print("[yellow]No data found.[/yellow]")
        return

    console.print(f"\n[bold]Dataset Statistics[/bold]")
    console.print(f"  Total results:    {len(df):,}")
    console.print(f"  Greyhounds:       {df['greyhound'].nunique():,}")
    console.print(f"  Date range:       {df['date'].min()} to {df['date'].max()}")

    if "track" in df.columns:
        console.print(f"  Tracks:           {df['track'].nunique()}")
        console.print(f"  Races:            {df['race_id'].nunique():,}" if "race_id" in df.columns else "")

    if "grade" in df.columns:
        console.print(f"  Grades:           {', '.join(sorted(df['grade'].dropna().unique()))}")

    # Track breakdown
    if "track" in df.columns:
        table = Table(title="Results by Track")
        table.add_column("Track")
        table.add_column("Results", justify="right")
        table.add_column("Date Range")

        for track, group in df.groupby("track"):
            table.add_row(
                str(track),
                f"{len(group):,}",
                f"{group['date'].min()} to {group['date'].max()}",
            )

        console.print(table)


@cli.command()
def elo_rankings():
    """Show current Elo ratings leaderboard."""
    from .pipeline import Pipeline

    pipeline = Pipeline()
    try:
        pipeline.load()
    except FileNotFoundError:
        console.print("[red]No trained model found. Run 'greyhound train' first.[/red]")
        return

    top = pipeline.elo.top_rated(30)

    table = Table(title="Elo Rating Leaderboard")
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Greyhound", style="white")
    table.add_column("Rating", style="green", width=8)

    for i, (name, rating) in enumerate(top, 1):
        table.add_row(str(i), name, f"{rating:.0f}")

    console.print(table)


if __name__ == "__main__":
    cli()
