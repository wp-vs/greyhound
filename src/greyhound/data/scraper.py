"""Historical data scraper for UK greyhound race results.

Uses the GBGB (Greyhound Board of Great Britain) JSON API at
api.gbgb.org.uk to extract race results. The API provides:

1. Paginated results listing: GET /api/results?page=N
   Returns meeting summaries with meeting IDs.

2. Meeting detail: GET /api/results/meeting/{meetingId}
   Returns full race details including traps/runners for a meeting.

The approach:
- Iterate through paginated results to discover meeting IDs
- Filter by date range and optionally by track
- Fetch full meeting details for each meeting
- Parse JSON into Race/RaceResult objects

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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from .schema import Race, RaceResult, UK_TRACKS

logger = logging.getLogger(__name__)

# Rate limiting — be respectful to avoid Cloudflare blocks
REQUEST_DELAY = 2.0  # seconds between requests


class GreyhoundScraper:
    """Scrapes historical UK greyhound race results via the GBGB JSON API."""

    API_BASE = "https://api.gbgb.org.uk/api"

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
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": "https://www.gbgb.org.uk/",
            "Origin": "https://www.gbgb.org.uk",
        })
        return session

    def _rate_limited_get(self, url: str, **kwargs) -> requests.Response:
        """GET request with rate limiting and retries."""
        elapsed = time_module.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time_module.sleep(REQUEST_DELAY - elapsed)

        for attempt in range(4):
            try:
                resp = self.session.get(url, timeout=30, **kwargs)
                self._last_request_time = time_module.time()
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                wait = 2 ** (attempt + 1)
                if attempt == 3:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}): {e} — retrying in {wait}s")
                time_module.sleep(wait)

        raise RuntimeError("Unreachable")

    def scrape_results(
        self,
        start_date: date,
        end_date: date,
        tracks: Optional[list[str]] = None,
    ) -> list[Race]:
        """Scrape race results for a date range via the GBGB API.

        Args:
            start_date: First date to scrape (inclusive).
            end_date: Last date to scrape (inclusive).
            tracks: List of track names to filter. None = all tracks.

        Returns:
            List of Race objects with results.
        """
        # Step 1: Discover meeting IDs by iterating through paginated results
        meeting_ids = self._discover_meetings(start_date, end_date, tracks)
        logger.info(f"Found {len(meeting_ids)} meetings to scrape")

        # Step 2: Fetch full details for each meeting
        all_races = []
        for i, (meeting_id, track, meeting_date) in enumerate(meeting_ids):
            logger.info(
                f"  [{i + 1}/{len(meeting_ids)}] Fetching meeting {meeting_id} "
                f"({track}, {meeting_date})"
            )
            try:
                races = self._fetch_meeting(meeting_id, track, meeting_date)
                all_races.extend(races)
                logger.info(f"    Got {len(races)} races")
            except Exception as e:
                logger.error(f"    Failed to fetch meeting {meeting_id}: {e}")

        logger.info(f"Total: {len(all_races)} races scraped")
        return all_races

    def _discover_meetings(
        self,
        start_date: date,
        end_date: date,
        tracks: Optional[list[str]] = None,
    ) -> list[tuple[int, str, date]]:
        """Discover meeting IDs by paginating through the results API.

        Returns list of (meeting_id, track_name, date) tuples.
        """
        meetings = []
        page = 1
        max_pages = 5000  # safety limit
        seen_ids = set()
        reached_start = False

        while page <= max_pages:
            logger.info(f"Scanning results page {page}...")
            url = f"{self.API_BASE}/results?page={page}"

            try:
                resp = self._rate_limited_get(url)
                data = resp.json()
            except Exception as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                break

            # Handle different response formats
            items = self._extract_items(data)

            if not items:
                logger.info(f"No more results at page {page}")
                break

            for item in items:
                meeting_id = self._get_field(item, ["meetingId", "meeting_id", "id", "Id"])
                track_name = self._get_field(
                    item, ["trackName", "track_name", "track", "Track", "venue", "Venue"]
                )
                date_str = self._get_field(
                    item, ["meetingDate", "meeting_date", "date", "Date", "raceDate"]
                )

                if meeting_id is None or date_str is None:
                    continue

                meeting_id = int(meeting_id)
                if meeting_id in seen_ids:
                    continue
                seen_ids.add(meeting_id)

                # Parse date
                meeting_date = self._parse_date(date_str)
                if meeting_date is None:
                    continue

                # Check date range
                if meeting_date > end_date:
                    continue
                if meeting_date < start_date:
                    reached_start = True
                    continue

                # Filter by track if specified
                if tracks and track_name:
                    if not any(t.lower() in track_name.lower() for t in tracks):
                        continue

                meetings.append((meeting_id, track_name or "Unknown", meeting_date))

            # If we've gone past the start date, stop paginating
            if reached_start:
                # Check if ALL items on this page are before start_date
                page_dates = []
                for item in items:
                    ds = self._get_field(
                        item, ["meetingDate", "meeting_date", "date", "Date", "raceDate"]
                    )
                    if ds:
                        d = self._parse_date(ds)
                        if d:
                            page_dates.append(d)
                if page_dates and all(d < start_date for d in page_dates):
                    logger.info("All results on page are before start_date, stopping")
                    break

            page += 1

        return sorted(meetings, key=lambda x: x[2])

    def _fetch_meeting(
        self, meeting_id: int, track: str, meeting_date: date
    ) -> list[Race]:
        """Fetch full race details for a single meeting."""
        url = f"{self.API_BASE}/results/meeting/{meeting_id}"
        params = {"meeting": meeting_id}

        resp = self._rate_limited_get(url, params=params)
        data = resp.json()

        return self._parse_meeting_response(data, meeting_id, track, meeting_date)

    def _parse_meeting_response(
        self, data: Any, meeting_id: int, track: str, meeting_date: date
    ) -> list[Race]:
        """Parse the JSON response from a meeting detail endpoint."""
        races = []

        # The response may be a list or a dict with a races key
        if isinstance(data, list):
            # Each item may be a meeting with races
            for meeting in data:
                race_list = self._get_field(meeting, ["races", "Races"]) or []
                for race_data in race_list:
                    race = self._parse_race(race_data, meeting_id, track, meeting_date)
                    if race and race.results:
                        races.append(race)
        elif isinstance(data, dict):
            race_list = self._get_field(data, ["races", "Races"]) or []
            if not race_list:
                # Maybe the dict itself contains race data at the top level
                race_list = [data]
            for race_data in race_list:
                race = self._parse_race(race_data, meeting_id, track, meeting_date)
                if race and race.results:
                    races.append(race)

        return races

    def _parse_race(
        self, race_data: dict, meeting_id: int, track: str, meeting_date: date
    ) -> Optional[Race]:
        """Parse a single race from the API response."""
        race_id_val = self._get_field(race_data, ["raceId", "race_id", "id", "Id"])
        race_number = self._get_field(
            race_data, ["raceNumber", "race_number", "raceNo", "RaceNo"]
        )
        distance = self._get_field(
            race_data, ["distance", "Distance", "raceDistance"]
        )
        grade = self._get_field(
            race_data, ["raceGrade", "grade", "Grade", "raceClass"]
        )
        race_type = self._get_field(
            race_data, ["raceType", "race_type", "type"]
        )
        going = self._get_field(
            race_data, ["going", "Going", "goingDescription"]
        )
        winning_time = self._get_field(
            race_data, ["winningTime", "winning_time", "winTime"]
        )
        race_time_str = self._get_field(
            race_data, ["raceTime", "race_time", "time", "offTime"]
        )
        prize = self._get_field(
            race_data, ["prizeMoney", "prize_money", "firstPrize", "totalPrize"]
        )
        forecast = self._get_field(
            race_data, ["forecast", "Forecast", "forecastDividend"]
        )
        tricast = self._get_field(
            race_data, ["tricast", "Tricast", "tricastDividend"]
        )

        # Build race ID
        race_id = f"{meeting_date.isoformat()}_{track}_{race_number or race_id_val or 0}"

        race = Race(
            race_id=race_id,
            date=meeting_date,
            track=track,
            race_number=int(race_number or 0),
            distance=int(distance or 0),
            grade=str(grade) if grade else None,
            race_type=str(race_type) if race_type else None,
            going=str(going) if going else None,
            winning_time=float(winning_time) if winning_time else None,
            prize_money=float(prize) if prize else None,
            forecast=str(forecast) if forecast else None,
            tricast=str(tricast) if tricast else None,
        )

        # Parse traps/runners
        traps = self._get_field(race_data, ["traps", "Traps", "runners", "Runners", "dogs"]) or []
        for trap_data in traps:
            result = self._parse_trap(trap_data)
            if result:
                race.results.append(result)

        # Sort results by position
        race.results.sort(key=lambda r: r.finish_position if r.finish_position else 99)

        return race

    def _parse_trap(self, trap_data: dict) -> Optional[RaceResult]:
        """Parse a single trap/runner from the API response."""
        dog_name = self._get_field(
            trap_data,
            ["dogName", "dog_name", "name", "Name", "greyhoundName", "dogname"],
        )
        if not dog_name:
            return None

        trap = self._get_field(
            trap_data, ["trapNumber", "trap_number", "trap", "Trap", "trapNo"]
        )
        position = self._get_field(
            trap_data,
            ["resultPosition", "result_position", "position", "Position",
             "finishPosition", "pos", "finishingPosition"],
        )
        finish_time = self._get_field(
            trap_data,
            ["resultRunTime", "run_time", "runTime", "time", "Time",
             "finishTime", "calcTime"],
        )
        sectional = self._get_field(
            trap_data,
            ["resultSectionalTime", "sectional_time", "sectionalTime",
             "sectional", "bendTime", "firstSectionalTime"],
        )
        sp = self._get_field(
            trap_data,
            ["resultStartingPrice", "starting_price", "startingPrice",
             "sp", "SP", "bsp"],
        )
        weight = self._get_field(
            trap_data, ["dogWeight", "weight", "Weight"]
        )
        trainer = self._get_field(
            trap_data, ["trainerName", "trainer_name", "trainer", "Trainer"]
        )
        comment = self._get_field(
            trap_data,
            ["resultComment", "comment", "Comment", "runComment"],
        )
        btn = self._get_field(
            trap_data,
            ["resultBtnDistance", "btn_distance", "btnDistance",
             "beatenDistance", "btn"],
        )

        # Parse position — handle non-finishers
        pos_int = self._parse_position(position)
        if pos_int is None:
            return None

        # Parse SP to decimal
        sp_decimal = None
        if sp is not None:
            sp_decimal = self._parse_sp(str(sp))

        return RaceResult(
            greyhound_name=str(dog_name).strip(),
            trap=int(trap) if trap else 0,
            finish_position=pos_int,
            finish_time=float(finish_time) if finish_time else None,
            sectional_time=float(sectional) if sectional else None,
            starting_price=str(sp) if sp else None,
            starting_price_decimal=sp_decimal,
            weight=float(weight) if weight else None,
            trainer=str(trainer) if trainer else None,
            comment=str(comment) if comment else None,
            btn=float(btn) if btn else None,
        )

    # --- Helper methods ---

    @staticmethod
    def _get_field(data: dict, field_names: list[str]) -> Any:
        """Try multiple field names and return the first match."""
        for name in field_names:
            if name in data:
                val = data[name]
                if val is not None and val != "":
                    return val
        return None

    @staticmethod
    def _extract_items(data: Any) -> list[dict]:
        """Extract the list of items from a paginated API response."""
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Try common pagination wrapper keys
            for key in ["data", "results", "items", "meetings", "Records",
                        "records", "Meetings", "Data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

            # Maybe the dict has numeric keys (paginated items)
            if "1" in data or 1 in data:
                return list(data.values())

        return []

    @staticmethod
    def _parse_date(date_str: Any) -> Optional[date]:
        """Parse various date formats to a date object."""
        if isinstance(date_str, date):
            return date_str
        if not isinstance(date_str, str):
            return None

        # Try ISO format first
        for fmt in [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d %b %Y",
            "%d %B %Y",
        ]:
            try:
                return datetime.strptime(date_str[:len(fmt) + 5], fmt).date()
            except (ValueError, IndexError):
                continue

        # Try parsing just the date part
        match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()

        return None

    @staticmethod
    def _parse_position(pos: Any) -> Optional[int]:
        """Parse finishing position, handling non-finishers."""
        if pos is None:
            return None
        if isinstance(pos, (int, float)):
            p = int(pos)
            return p if p >= 1 else None

        pos_str = str(pos).strip().upper()
        # Non-finishers
        if pos_str in ("DNF", "NR", "DNS", "FO", "RO", "BD", ""):
            return None

        match = re.search(r"(\d+)", pos_str)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _parse_sp(text: str) -> Optional[float]:
        """Parse starting price to decimal odds."""
        text = text.strip()

        # Evens
        if text.lower() in ("evs", "evens", "ev"):
            return 2.0

        # Fractional: "5/2", "3/1", "11/4"
        match = re.search(r"(\d+)/(\d+)", text)
        if match:
            num, den = int(match.group(1)), int(match.group(2))
            if den > 0:
                return round(num / den + 1, 2)

        # Already decimal: "3.50", "2.10"
        try:
            val = float(text)
            if 1.01 <= val <= 200.0:
                return val
        except ValueError:
            pass

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


class BetfairBSPLoader:
    """Download and parse Betfair BSP (Starting Price) CSV files.

    Betfair publishes free daily CSVs at promo.betfair.com with BSP odds
    and win/lose outcomes for every greyhound runner. This provides
    historical data going back to ~2018.

    CSV columns: EVENT_DT, EVENT_ID, MENU_HINT, EVENT_NAME,
                 SELECTION_NAME, WIN_LOSE, BSP, PPWAP, PPMAX, PPMIN,
                 IPMAX, IPMIN, PPTRADEDVOL, IPTRADEDVOL
    """

    BASE_URL = "https://promo.betfair.com/betfairsp/prices"
    WIN_PATTERN = "dwbfgreyhoundwin{date}.csv"
    PLACE_PATTERN = "dwbfgreyhoundplace{date}.csv"

    def __init__(
        self,
        output_dir: str = "data/raw/betfair_bsp",
        session: Optional[requests.Session] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or self._create_session()
        self._last_request_time = 0.0

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        return session

    def _rate_limited_get(self, url: str) -> Optional[requests.Response]:
        """GET with rate limiting and retries. Returns None on 404."""
        elapsed = time_module.time() - self._last_request_time
        if elapsed < 1.0:
            time_module.sleep(1.0 - elapsed)

        for attempt in range(4):
            try:
                resp = self.session.get(url, timeout=30)
                self._last_request_time = time_module.time()
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                wait = 2 ** (attempt + 1)
                if attempt == 3:
                    logger.error(f"Failed after 4 attempts: {url} — {e}")
                    return None
                logger.warning(f"Request failed (attempt {attempt + 1}): {e} — retrying in {wait}s")
                time_module.sleep(wait)
        return None

    def download_range(
        self,
        start_date: date,
        end_date: date,
        market: str = "win",
        skip_existing: bool = True,
    ) -> list[Path]:
        """Download daily BSP CSVs for a date range.

        Args:
            start_date: First date (inclusive).
            end_date: Last date (inclusive).
            market: 'win' or 'place'.
            skip_existing: Skip files already downloaded.

        Returns:
            List of paths to downloaded CSV files.
        """
        pattern = self.WIN_PATTERN if market == "win" else self.PLACE_PATTERN
        downloaded = []
        current = start_date
        total_days = (end_date - start_date).days + 1

        while current <= end_date:
            day_num = (current - start_date).days + 1
            date_str = current.strftime("%d%m%Y")
            filename = pattern.format(date=date_str)
            local_path = self.output_dir / filename

            if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
                logger.debug(f"  Skipping {filename} (already exists)")
                downloaded.append(local_path)
                current += timedelta(days=1)
                continue

            url = f"{self.BASE_URL}/{filename}"
            logger.info(f"  [{day_num}/{total_days}] Downloading {filename}...")
            resp = self._rate_limited_get(url)

            if resp is not None and resp.text.strip():
                local_path.write_text(resp.text, encoding="utf-8")
                downloaded.append(local_path)
                logger.info(f"    Saved ({len(resp.text):,} bytes)")
            else:
                logger.warning(f"    No data for {current} (no racing or file missing)")

            current += timedelta(days=1)

        logger.info(f"Downloaded {len(downloaded)} files to {self.output_dir}")
        return downloaded

    def parse_csv(self, filepath: Path) -> pd.DataFrame:
        """Parse a single Betfair BSP CSV into a standardised DataFrame."""
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to parse {filepath}: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Standardise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Filter out rows with no BSP
        if "BSP" in df.columns:
            df = df[pd.to_numeric(df["BSP"], errors="coerce").notna()].copy()
            df["BSP"] = pd.to_numeric(df["BSP"], errors="coerce")

        if df.empty:
            return df

        # Parse date
        if "EVENT_DT" in df.columns:
            df["date"] = pd.to_datetime(df["EVENT_DT"], dayfirst=True, errors="coerce").dt.date

        # Parse track from MENU_HINT (e.g. "Greyhounds - Romford" → "Romford")
        if "MENU_HINT" in df.columns:
            df["track"] = (
                df["MENU_HINT"]
                .astype(str)
                .str.replace(r"(?i)^.*?-\s*", "", regex=True)
                .str.strip()
            )

        # Parse SELECTION_NAME: "1. Dog Name" → trap=1, greyhound="Dog Name"
        if "SELECTION_NAME" in df.columns:
            sel = df["SELECTION_NAME"].astype(str)
            trap_match = sel.str.extract(r"^(\d+)\.\s*(.*)", expand=True)
            df["trap"] = pd.to_numeric(trap_match[0], errors="coerce").fillna(0).astype(int)
            df["greyhound"] = trap_match[1].str.strip().fillna(sel.str.strip())

        # Parse EVENT_NAME for race number and distance
        if "EVENT_NAME" in df.columns:
            event = df["EVENT_NAME"].astype(str)
            # Try to extract race number (e.g., "R1", "Race 1")
            race_num = event.str.extract(r"(?:R|Race\s*)(\d+)", expand=False)
            df["race_number"] = pd.to_numeric(race_num, errors="coerce").fillna(0).astype(int)
            # Try to extract distance in metres
            dist = event.str.extract(r"(\d{3,4})m", expand=False)
            df["distance"] = pd.to_numeric(dist, errors="coerce").fillna(0).astype(int)

        # Win/lose: 1=win, 0=lose, 2=dead heat
        if "WIN_LOSE" in df.columns:
            df["bsp_win"] = df["WIN_LOSE"].astype(str).str.strip()
            df["bsp_win"] = pd.to_numeric(df["bsp_win"], errors="coerce").fillna(0).astype(int)

        # Rename BSP
        if "BSP" in df.columns:
            df["bsp_decimal"] = df["BSP"]

        # Keep betfair event ID
        if "EVENT_ID" in df.columns:
            df["betfair_event_id"] = df["EVENT_ID"]

        # Select output columns
        out_cols = [
            "date", "track", "race_number", "distance", "trap",
            "greyhound", "bsp_decimal", "bsp_win", "betfair_event_id",
        ]
        out_cols = [c for c in out_cols if c in df.columns]
        return df[out_cols].reset_index(drop=True)

    def load_all(self) -> pd.DataFrame:
        """Load and combine all downloaded BSP CSVs."""
        frames = []
        csv_files = sorted(self.output_dir.glob("dwbfgreyhound*.csv"))
        for f in csv_files:
            df = self.parse_csv(f)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "track", "trap", "greyhound"], keep="last"
        )
        combined = combined.sort_values(["date", "track", "race_number", "trap"])
        combined = combined.reset_index(drop=True)
        logger.info(f"Loaded {len(combined)} BSP records from {len(csv_files)} files")
        return combined

    def save_parquet(self, df: pd.DataFrame, filename: str = "bsp_data.parquet") -> Path:
        """Save combined BSP data to parquet."""
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} BSP rows to {path}")
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

        # Load BSP data
        bsp_dir = self.data_dir / "raw" / "betfair_bsp"
        if bsp_dir.exists():
            bsp_loader = BetfairBSPLoader(output_dir=str(bsp_dir))
            bsp_df = bsp_loader.load_all()
            if not bsp_df.empty:
                frames.append(bsp_df)
                logger.info(f"Loaded {len(bsp_df)} BSP records")

        if not frames:
            logger.warning("No data files found")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "track", "race_number", "trap"], keep="last")
        df = df.sort_values(["date", "track", "race_number", "finish_position"])
        df = df.reset_index(drop=True)

        logger.info(f"Loaded {len(df)} total results")
        return df
