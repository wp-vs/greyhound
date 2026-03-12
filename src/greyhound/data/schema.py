"""Data schema definitions for greyhound race data."""

from dataclasses import dataclass, field
from datetime import date, time
from typing import Optional


# Standard UK greyhound tracks
UK_TRACKS = [
    "Belle Vue", "Central Park", "Crayford", "Doncaster", "Hall Green",
    "Harlow", "Hove", "Kinsley", "Monmore", "Newcastle",
    "Nottingham", "Oxford", "Pelaw Grange", "Perry Barr", "Poole",
    "Romford", "Sheffield", "Shelbourne Park", "Sunderland",
    "Swindon", "Towcester", "Wimbledon", "Yarmouth",
]

# Standard race distances in metres
STANDARD_DISTANCES = [210, 238, 250, 265, 270, 277, 285, 380, 400, 415, 430,
                      450, 460, 470, 480, 500, 515, 520, 550, 575, 590, 630,
                      640, 660, 680, 695, 710, 730, 750, 840, 870, 900, 925, 1000]

# Race grades (highest to lowest quality)
RACE_GRADES = ["OR", "S", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
               "A9", "A10", "A11", "B1", "B2", "B3", "B4", "B5", "B6",
               "D1", "D2", "D3", "D4", "E1", "E2", "IT", "P"]

GRADE_NUMERIC = {g: i for i, g in enumerate(RACE_GRADES)}


@dataclass
class RaceResult:
    """A single greyhound's result in a race."""
    greyhound_name: str
    trap: int  # 1-6 (or 1-8 for wide tracks)
    finish_position: int
    finish_time: Optional[float] = None  # seconds
    sectional_time: Optional[float] = None  # time to first bend
    starting_price: Optional[str] = None  # e.g. "5/2", "3/1"
    starting_price_decimal: Optional[float] = None
    weight: Optional[float] = None  # kg
    trainer: Optional[str] = None
    comment: Optional[str] = None  # race comment
    btn: Optional[float] = None  # beaten distance (lengths behind winner)


@dataclass
class Race:
    """A complete greyhound race."""
    race_id: str
    date: date
    track: str
    race_number: int
    distance: int  # metres
    grade: Optional[str] = None
    race_type: Optional[str] = None  # flat, hurdles
    going: Optional[str] = None  # track condition
    prize_money: Optional[float] = None
    race_time: Optional[time] = None
    winning_time: Optional[float] = None
    forecast: Optional[str] = None  # forecast dividend
    tricast: Optional[str] = None  # tricast dividend
    results: list[RaceResult] = field(default_factory=list)


@dataclass
class GreyhoundProfile:
    """Profile information for a greyhound."""
    name: str
    sire: Optional[str] = None
    dam: Optional[str] = None
    colour: Optional[str] = None
    sex: Optional[str] = None  # D=dog, B=bitch
    birth_date: Optional[date] = None
    trainer: Optional[str] = None
    owner: Optional[str] = None
    season_best: Optional[str] = None
