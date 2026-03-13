"""Fetch Timeform profiles for all Romford runners by finding their IDs
from recent results pages, then scraping their individual form pages."""

import re
import time
from datetime import date, timedelta

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

scraper = cloudscraper.create_scraper()

# Load our racecard
rc = pd.read_csv("data/romford_racecard_full_2026-03-13.csv")
our_dogs = set(rc["greyhound"].str.strip().str.upper())
print(f"Looking for {len(our_dogs)} dogs")

# Step 1: Find Timeform IDs by scanning recent results pages
dog_ids = {}  # name -> {slug, id}

# Scan last 14 days of results
start = date(2026, 2, 27)
end = date(2026, 3, 13)
d = start
while d <= end:
    url = f"https://www.timeform.com/greyhound-racing/results/{d}"
    r = scraper.get(url)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            match = re.match(r".*/greyhound-form/([^/]+)/(\d+)", href)
            if match:
                name = a.get_text(strip=True).upper()
                if name in our_dogs and name not in dog_ids:
                    dog_ids[name] = {"slug": match.group(1), "id": match.group(2)}
        found = len(dog_ids)
        remaining = len(our_dogs) - found
        print(f"  {d}: found {found}/{len(our_dogs)} dogs ({remaining} remaining)")
        if remaining == 0:
            break
    else:
        print(f"  {d}: status {r.status_code}")
    d += timedelta(days=1)
    time.sleep(2)

# Also check the Romford racecard page on Timeform for today
url = "https://www.timeform.com/greyhound-racing/racecards/romford/2026-03-13"
r = scraper.get(url)
if r.status_code == 200:
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        match = re.match(r".*/greyhound-form/([^/]+)/(\d+)", href)
        if match:
            name = a.get_text(strip=True).upper()
            if name in our_dogs and name not in dog_ids:
                dog_ids[name] = {"slug": match.group(1), "id": match.group(2)}
    print(f"  Racecard page: now have {len(dog_ids)}/{len(our_dogs)} dogs")

print(f"\nFound IDs for {len(dog_ids)}/{len(our_dogs)} dogs")
missing = our_dogs - set(dog_ids.keys())
if missing:
    print(f"Missing: {', '.join(sorted(missing))}")

# Step 2: Fetch each dog's full profile
all_history = []

for name, info in sorted(dog_ids.items()):
    url = f"https://www.timeform.com/greyhound-racing/greyhound-form/{info['slug']}/{info['id']}"
    r = scraper.get(url)
    if r.status_code != 200:
        print(f"  {name}: FAILED ({r.status_code})")
        time.sleep(2)
        continue

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    # Parse the results table
    # Pattern in Timeform profile: date, type, dist, proxy, TFSec, bend, TFGoing, TFTime
    # The data appears as lines with dates followed by race data
    lines = text.split("\n")

    runs = []
    i = 0
    while i < len(lines):
        # Look for date pattern DD/MM/YYYY
        dm = re.match(r"(\d{2}/\d{2}/\d{4})", lines[i].strip())
        if dm:
            date_str = dm.group(1)
            # Collect the next few lines as race data
            race_data = [lines[i]]
            j = i + 1
            while j < min(i + 15, len(lines)):
                if re.match(r"\d{2}/\d{2}/\d{4}", lines[j].strip()):
                    break
                race_data.append(lines[j].strip())
                j += 1

            race_text = " ".join(race_data)

            # Try to extract: position (from form like 1222), time, distance, comment
            # The format varies but typically includes:
            # date | track | dist | form/pos | time | comment
            run = {"greyhound": name, "date": date_str, "raw": race_text[:200]}
            runs.append(run)
            i = j
        else:
            i += 1

    n_runs = len(runs)
    print(f"  {name}: {n_runs} runs found")
    all_history.extend(runs)
    time.sleep(1.5)

# Save raw data
hist_df = pd.DataFrame(all_history)
hist_df.to_csv("data/romford_timeform_profiles_2026-03-13.csv", index=False)
print(f"\nSaved {len(all_history)} records to data/romford_timeform_profiles_2026-03-13.csv")
