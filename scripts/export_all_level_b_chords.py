"""
Driver: generate chord HTMLs for all Level B sectors sequentially.
Each sector runs in its own process (export_chord_level_b.py), which
avoids memory accumulation at the cost of re-loading the trade parquet
per sector.

Usage: python export_all_level_b_chords.py [--top-n 15]
"""
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top-n", type=int, default=15)
args = parser.parse_args()

# All 8 sector pairs (electronics already generated; still refreshed for consistency)
SECTORS = ["electronics", "vehicles", "pharma", "machinery",
           "oil_fuels", "iron_steel", "plastics", "apparel"]

for i, sector in enumerate(SECTORS, 1):
    print(f"\n{'='*60}\n[{i}/8] {sector}\n{'='*60}", flush=True)
    r = subprocess.run(
        [sys.executable, "export_chord_level_b.py",
         "--sector", sector, "--top-n", str(args.top_n)])
    if r.returncode != 0:
        print(f"  WARNING: {sector} exited with code {r.returncode}", flush=True)

print("\nAll sector chords done.")
