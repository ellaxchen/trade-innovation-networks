"""
Application-level inventor country from PATSTAT person tables (tls206/tls207).

Rule: inventor country = person_ctry_code of the FIRST-LISTED inventor
(lowest invt_seq_nr) whose person record carries a non-blank country code.
Persons with blank country codes are treated as missing (they are ~34% of
person records in this edition). Applications with no inventor record that
has a country get no entry (dropped).

This is the accurate counterpart to appln_auth (filing office) for the
country-assignment question flagged throughout the project.

Output: ../processed/appln_inventor_country.parquet
        (appln_id, inventor_country)

Usage: python build_inventor_country.py
"""
import pandas as pd
import pyarrow.parquet as pq

print("[1/3] Loading tls206 person countries...")
persons = pq.read_table(
    "../rawdata/tls206.parquet", columns=["person_id", "person_ctry_code"]
).to_pandas()
# This edition uses a two-space placeholder for missing country codes,
# so require a genuine two-letter alphabetic code.
persons["person_ctry_code"] = persons["person_ctry_code"].astype(str).str.strip()
persons = persons[
    persons["person_ctry_code"].str.fullmatch(r"[A-Za-z]{2}").fillna(False)
]
persons["person_ctry_code"] = persons["person_ctry_code"].astype("category")
persons = persons.drop_duplicates(subset="person_id")
person_map = persons.set_index("person_id")["person_ctry_code"]
del persons
print(f"  {len(person_map):,} persons with a non-blank country code")

print("[2/3] Streaming tls207 inventor links...")
pf = pq.ParquetFile("../rawdata/tls207.parquet")
parts = []
total_inv_rows = 0
mapped_rows = 0
for i, batch in enumerate(pf.iter_batches(
        batch_size=5_000_000, columns=["appln_id", "person_id", "invt_seq_nr"])):
    b = batch.to_pandas()
    b = b[b["invt_seq_nr"] > 0]
    total_inv_rows += len(b)
    if len(b) == 0:
        continue
    b["inventor_country"] = b["person_id"].map(person_map)
    b = b.dropna(subset=["inventor_country"])
    mapped_rows += len(b)
    if len(b) == 0:
        continue
    # keep the first-listed inventor with a known country per application
    b = b.sort_values(["appln_id", "invt_seq_nr"]).drop_duplicates(
        subset="appln_id", keep="first")
    parts.append(b[["appln_id", "invt_seq_nr", "inventor_country"]])
    if (i + 1) % 10 == 0:
        print(f"  {(i+1)*5:>4}M rows scanned, {mapped_rows:,} inventor rows mapped")

print(f"  inventor rows: {total_inv_rows:,}; with a country: {mapped_rows:,} "
      f"({mapped_rows/total_inv_rows*100:.1f}%)")

print("[3/3] Collapsing to one country per application...")
out = pd.concat(parts, ignore_index=True)
del parts
out = out.sort_values(["appln_id", "invt_seq_nr"]).drop_duplicates(
    subset="appln_id", keep="first")
out = out[["appln_id", "inventor_country"]].copy()
out["inventor_country"] = out["inventor_country"].astype(str).astype("string")
out = out.sort_values("appln_id")
out.to_parquet("../processed/appln_inventor_country.parquet", index=False)
print(f"  Saved {len(out):,} applications -> ../processed/appln_inventor_country.parquet")
print(f"  Distinct country codes: {out['inventor_country'].nunique():,}")
print(out["inventor_country"].value_counts().head(15).to_string())
print("\nDone.")
