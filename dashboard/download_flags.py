"""
download_flags.py — One-time script to download WC2026 flag PNGs locally.

Run this once from the project root:
    python dashboard/download_flags.py

Flags are saved to dashboard/flags/<iso2>.png  (w80 resolution)
"""
import urllib.request
from pathlib import Path

FLAGS_DIR = Path(__file__).parent / "flags"
FLAGS_DIR.mkdir(exist_ok=True)

TEAM_ISO2 = {
    "Mexico": "mx",
    "South Africa": "za",
    "South Korea": "kr",
    "Czechia": "cz",
    "Canada": "ca",
    "Switzerland": "ch",
    "Qatar": "qa",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br",
    "Morocco": "ma",
    "Haiti": "ht",
    "Scotland": "gb-sct",
    "USA": "us",
    "Paraguay": "py",
    "Australia": "au",
    "Türkiye": "tr",
    "Germany": "de",
    "Curaçao": "cw",
    "Ivory Coast": "ci",
    "Ecuador": "ec",
    "Netherlands": "nl",
    "Japan": "jp",
    "Tunisia": "tn",
    "Sweden": "se",
    "Belgium": "be",
    "Egypt": "eg",
    "Iran": "ir",
    "New Zealand": "nz",
    "Spain": "es",
    "Cape Verde": "cv",
    "Saudi Arabia": "sa",
    "Uruguay": "uy",
    "France": "fr",
    "Senegal": "sn",
    "Norway": "no",
    "Iraq": "iq",
    "Argentina": "ar",
    "Algeria": "dz",
    "Austria": "at",
    "Jordan": "jo",
    "Portugal": "pt",
    "Uzbekistan": "uz",
    "Colombia": "co",
    "DR Congo": "cd",
    "England": "gb-eng",
    "Croatia": "hr",
    "Ghana": "gh",
    "Panama": "pa",
}

def download_all():
    ok, fail = 0, 0
    for team, iso2 in TEAM_ISO2.items():
        path = FLAGS_DIR / f"{iso2}.png"
        if path.exists():
            print(f"  [skip]  {team:30s}  {iso2}.png  (already exists)")
            ok += 1
            continue
        url = f"https://flagcdn.com/w80/{iso2}.png"
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  [ok]    {team:30s}  {iso2}.png")
            ok += 1
        except Exception as e:
            print(f"  [FAIL]  {team:30s}  {iso2}.png  -- {e}")
            fail += 1
    print(f"\nDone: {ok} ok, {fail} failed.")

if __name__ == "__main__":
    download_all()
