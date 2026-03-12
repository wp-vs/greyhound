"""Historical data scraper for UK greyhound race results.

Extracts race data from publicly available sources including:
- GBGB (Greyhound Board of Great Britain) results
- SIS/timeform data where available
- Historical results archives

Usage:
    scraper = GreyhoundScraper(output_dir="data/raw")
    races = scraper.scrape_results(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        tracks=["Romford", "Crayford"]
    )
"""

import logging
import re
import time as time_module
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .schema import Race, RaceResult, UK_TRACKS

logger = logging.getLogger(__name__)

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests


class GreyhoundScraper:
    """Scrapes historical UK greyhound race results."""

    # GBGB results base URL
    GBGB_BASE = "https://www.gbgb.org.uk"
    GBGB_RESULTS = f"{GBGB_BASE}/results-archive/"

    def __init__(self, output_dir: str = "data/raw", session: Optional[requests.Session] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or self._create_session()
        self._last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        })
        return session

    def _rate_limited_get(self, url: str, **kwargs) -> requests.Response:
        """GET request with rate limiting and retries."""
        elapsed = time_module.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time_module.sleep(REQUEST_DELAY - elapsed)

        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=30, **kwargs)
                self._last_request_time = time_module.time()
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time_module.sleep(2 ** (attempt + 1))

        raise RuntimeError("Unreachable")

    def scrape_results(
        self,
        start_date: date,
        end_date: date,
        tracks: Optional[list[str]] = None,
    ) -> list[Race]:
        """Scrape race results for a date range.

        Args:
            start_date: First date to scrape (inclusive).
            end_date: Last date to scrape (inclusive).
            tracks: List of track names to filter. None = all tracks.

        Returns:
            List of Race objects with results.
        """
        all_races = []
        current = start_date

        while current <= end_date:
            logger.info(f"Scraping results for {current}")
            try:
                day_races = self._scrape_day(current, tracks)
                all_races.extend(day_races)
                logger.info(f"  Found {len(day_races)} races")
            except Exception as e:
                logger.error(f"  Failed to scrape {current}: {e}")

            current += timedelta(days=1)

        return all_races

    def _scrape_day(self, race_date: date, tracks: Optional[list[str]] = None) -> list[Race]:
        """Scrape all races for a single day from GBGB."""
        date_str = race_date.strftime("%Y-%m-%d")
        url = f"{self.GBGB_RESULTS}?searchDate={date_str}"

        resp = self._rate_limited_get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        races = []
        # Find meeting links for the day
        meeting_links = self._extract_meeting_links(soup, tracks)

        for track_name, meeting_url in meeting_links:
            try:
                meeting_races = self._scrape_meeting(track_name, meeting_url, race_date)
                races.extend(meeting_races)
            except Exception as e:
                logger.error(f"  Failed to scrape meeting {track_name}: {e}")

        return races

    def _extract_meeting_links(
        self, soup: BeautifulSoup, tracks: Optional[list[str]] = None
    ) -> list[tuple[str, str]]:
        """Extract meeting links from the results archive page."""
        links = []
        for link in soup.select("a[href*='results']"):
            text = link.get_text(strip=True)
            href = link.get("href", "")

            # Match track names
            for track in (tracks or UK_TRACKS):
                if track.lower() in text.lower():
                    full_url = urljoin(self.GBGB_BASE, href)
                    links.append((track, full_url))
                    break

        return links

    def _scrape_meeting(
        self, track: str, url: str, race_date: date
    ) -> list[Race]:
        """Scrape all races from a single meeting page."""
        resp = self._rate_limited_get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        races = []
        race_sections = soup.select(".race-result, .raceResult, [class*='race']")

        for i, section in enumerate(race_sections, 1):
            try:
                race = self._parse_race_section(section, track, race_date, i)
                if race and race.results:
                    races.append(race)
            except Exception as e:
                logger.warning(f"  Failed to parse race {i} at {track}: {e}")

        return races

    def _parse_race_section(
        self, section: BeautifulSoup, track: str, race_date: date, race_num: int
    ) -> Optional[Race]:
        """Parse a single race result section."""
        # Extract race metadata
        header = section.find(["h2", "h3", "h4", ".race-header", "[class*='header']"])
        header_text = header.get_text(strip=True) if header else ""

        distance = self._extract_distance(header_text)
        grade = self._extract_grade(header_text)

        race_id = f"{race_date.isoformat()}_{track}_{race_num}"

        race = Race(
            race_id=race_id,
            date=race_date,
            track=track,
            race_number=race_num,
            distance=distance or 0,
            grade=grade,
        )

        # Extract individual results
        rows = section.select("tr, .runner, [class*='runner']")
        for row in rows:
            result = self._parse_result_row(row)
            if result:
                race.results.append(result)

        return race

    def _parse_result_row(self, row: BeautifulSoup) -> Optional[RaceResult]:
        """Parse a single runner result row."""
        cells = row.find_all(["td", "span", "div"])
        if len(cells) < 3:
            return None

        texts = [c.get_text(strip=True) for c in cells]

        # Try to extract position, trap, name, time, SP
        position = self._extract_int(texts[0]) if texts else None
        if position is None or position < 1:
            return None

        trap = self._extract_int(texts[1]) if len(texts) > 1 else None
        name = texts[2] if len(texts) > 2 else "Unknown"

        # Clean greyhound name
        name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
        if not name or len(name) < 2:
            return None

        # Extract finishing time
        finish_time = None
        sp_decimal = None
        for t in texts[3:]:
            if not finish_time:
                finish_time = self._extract_time(t)
            if not sp_decimal:
                sp_decimal = self._parse_sp(t)

        return RaceResult(
            greyhound_name=name,
            trap=trap or 0,
            finish_position=position,
            finish_time=finish_time,
            starting_price_decimal=sp_decimal,
        )

    @staticmethod
    def _extract_distance(text: str) -> Optional[int]:
        """Extract race distance in metres from text."""
        match = re.search(r"(\d{3,4})\s*m", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Try yards conversion
        match = re.search(r"(\d{3,4})\s*y", text, re.IGNORECASE)
        if match:
            return int(int(match.group(1)) * 0.9144)
        return None

    @staticmethod
    def _extract_grade(text: str) -> Optional[str]:
        """Extract race grade from text."""
        match = re.search(r"\b(OR|S|[A-E]\d{1,2}|IT|P|D\d)\b", text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_int(text: str) -> Optional[int]:
        """Extract first integer from text."""
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_time(text: str) -> Optional[float]:
        """Extract time in seconds from text like '29.45' or '30.12'."""
        match = re.search(r"(\d{2,3}\.\d{1,2})", text)
        if match:
            val = float(match.group(1))
            if 15.0 < val < 120.0:  # reasonable greyhound race time range
                return val
        return None

    @staticmethod
    def _parse_sp(text: str) -> Optional[float]:
        """Parse starting price to decimal odds."""
        # Fractional: "5/2", "3/1", "11/4"
        match = re.search(r"(\d+)/(\d+)", text)
        if match:
            num, den = int(match.group(1)), int(match.group(2))
            if den > 0:
                return round(num / den + 1, 2)
        # Decimal: "3.50", "2.10"
        match = re.search(r"^(\d+\.\d{1,2})$", text.strip())
        if match:
            val = float(match.group(1))
            if 1.01 <= val <= 100.0:
                return val
        # Evens
        if text.strip().lower() in ("evs", "evens"):
            return 2.0
        return None

    def to_dataframe(self, races: list[Race]) -> pd.DataFrame:
        """Convert list of Race objects to a flat DataFrame."""
        rows = []
        for race in races:
            for result in race.results:
                rows.append({
                    "race_id": race.race_id,
                    "date": race.date,
                    "track": race.track,
                    "race_number": race.race_number,
                    "distance": race.distance,
                    "grade": race.grade,
                    "race_type": race.race_type,
                    "going": race.going,
                    "winning_time": race.winning_time,
                    "greyhound": result.greyhound_name,
                    "trap": result.trap,
                    "finish_position": result.finish_position,
                    "finish_time": result.finish_time,
                    "sectional_time": result.sectional_time,
                    "sp_decimal": result.starting_price_decimal,
                    "weight": result.weight,
                    "trainer": result.trainer,
                    "btn": result.btn,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["date", "track", "race_number", "finish_position"])
            df = df.reset_index(drop=True)
        return df

    def save_results(self, races: list[Race], filename: str = "results.parquet") -> Path:
        """Save race results to parquet file."""
        df = self.to_dataframe(races)
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path


class SISDataLoader:
    """Load and parse SIS greyhound data feeds.

    SIS (Satellite Information Services) provides official data feeds
    for UK greyhound racing. This loader handles their CSV/XML formats.
    """

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load a SIS-format CSV results file."""
        df = pd.read_csv(filepath)

        # Standardise column names
        col_map = {
            "Track": "track",
            "Date": "date",
            "Race": "race_number",
            "Dist": "distance",
            "Grade": "grade",
            "Dog": "greyhound",
            "Trap": "trap",
            "Pos": "finish_position",
            "Time": "finish_time",
            "SP": "sp_decimal",
            "Wgt": "weight",
            "Trainer": "trainer",
            "Going": "going",
            "Btn": "btn",
            "Sect": "sectional_time",
        }

        renamed = {}
        for old, new in col_map.items():
            matches = [c for c in df.columns if old.lower() in c.lower()]
            if matches:
                renamed[matches[0]] = new

        df = df.rename(columns=renamed)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)

        return df


class HistoricalDataLoader:
    """Load historical data from multiple sources and merge."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> pd.DataFrame:
        """Load and merge all available data sources."""
        frames = []

        # Load parquet files
        for f in (self.data_dir / "raw").glob("*.parquet"):
            logger.info(f"Loading {f}")
            frames.append(pd.read_parquet(f))

        # Load CSV files
        for f in (self.data_dir / "raw").glob("*.csv"):
            logger.info(f"Loading {f}")
            try:
                frames.append(SISDataLoader.load_csv(str(f)))
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if not frames:
            logger.warning("No data files found")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "track", "race_number", "trap"], keep="last")
        df = df.sort_values(["date", "track", "race_number", "finish_position"])
        df = df.reset_index(drop=True)

        logger.info(f"Loaded {len(df)} total results")
        return df
