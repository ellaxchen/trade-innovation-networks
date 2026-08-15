"""
Export chord data as JSON for D3. Interleaved matrix with reverse entries.
"""
import pandas as pd; import numpy as np; import json; import argparse; import os

parser = argparse.ArgumentParser()
parser.add_argument("--top-n", type=int, default=21)
parser.add_argument("--output", type=str, default="../html/chord_data.json")
args = parser.parse_args()
TOP_N = args.top_n
INDIV_N = TOP_N - 1

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"])); baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"])); iso_to_name["TW"] = "Taiwan"

# Clean historical/verbose country names for display
CLEAN_NAMES = {
    "Fed. Rep. of Germany (...1990)": "Germany",
    "Belgium-Luxembourg (...1998)": "Belgium",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Mayotte (Overseas France)": "Mayotte",
    "Falkland Isds (Malvinas)": "Falkland Islands",
    "Russian Federation": "Russia",
    "Netherlands Antilles (...2010)": "Netherlands Antilles",
    "Sudan (...2011)": "Sudan",
    "Dem. Rep. of Germany (...1990)": "East Germany",
    "Czechoslovakia (...1992)": "Czechoslovakia",
    "USSR (...1990)": "USSR",
    "Serbia and Montenegro (...2005)": "Serbia and Montenegro",
    "Southern African Customs Union (...1999)": "S. African Customs Union",
}
for old, new in CLEAN_NAMES.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

trade_raw = pd.read_parquet("../rawdata/baci_panel_hs6.parquet",
                            columns=["year","exporter","importer","trade_value"])
trade_raw["source"] = trade_raw["exporter"].map(baci_to_iso)
trade_raw["target"] = trade_raw["importer"].map(baci_to_iso)
trade_raw = trade_raw.dropna(subset=["source","target"])
trade_raw = trade_raw[trade_raw["source"] != trade_raw["target"]]
trade_raw = trade_raw.groupby(["year","source","target"], as_index=False)["trade_value"].sum()

cite_raw = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite_raw = cite_raw.rename(columns={"citing_country":"source","cited_country":"target","citation_count":"weight"})
NON = {"EP","WO","DD","OA","AP","GC","IB","EM","BX","YU","CS","SU"}
cite_raw = cite_raw[(~cite_raw["source"].isin(NON)) & (~cite_raw["target"].isin(NON))]
cite_raw = cite_raw[cite_raw["source"] != cite_raw["target"]]

def tot_flow(df, wcol):
    s = df.groupby("source")[wcol].sum(); r = df.groupby("target")[wcol].sum()
    return s.add(r, fill_value=0)

trade_vol = tot_flow(trade_raw, "trade_value")
cite_vol = tot_flow(cite_raw, "weight")
combined = trade_vol.add(cite_vol, fill_value=0)
top_individuals = combined.nlargest(INDIV_N).index.tolist()
all_countries = top_individuals + ["ROW"]
print(f"Countries: {all_countries}")

COLORS = ["#E63946","#3A7CA5","#2A9D8F","#F4A261","#9B5DE5","#E76F51",
          "#264653","#A8DADC","#E9C46A","#6D597A","#B5838D","#4895EF",
          "#FF6B6B","#4ECDC4","#FFD166","#F15BB5","#00BBF9","#FEE440",
          "#D4A373","#073B4C"]
node_colors = {}
for i, n in enumerate(top_individuals):
    node_colors[n] = COLORS[i % len(COLORS)]
node_colors["ROW"] = "#999999"

def build_matrices(df, wcol, years, individuals, all_groups):
    result = {}
    ind_set = set(individuals)
    N = len(all_groups)
    for yr in years:
        yr_df = df[df["year"] == yr]
        idx = {c: i for i, c in enumerate(all_groups)}
        # Build: rows 0..N-1=exports, N..2N-1=imports
        mat = [[0.0] * (2*N) for _ in range(2*N)]

        for _, r in yr_df.iterrows():
            s, t = r["source"], r["target"]
            v = float(r[wcol])
            if v <= 0 or s == t: continue
            s_grp = s if s in ind_set else "ROW"
            t_grp = t if t in ind_set else "ROW"
            if s_grp == t_grp: continue
            i_src = idx[s_grp]
            i_tgt = idx[t_grp] + N
            mat[i_src][i_tgt] += v

        # Compute correct arc sizes: exports from row sums, imports from column sums
        exp_sizes = [float(sum(mat[i])) for i in range(N)]
        imp_sizes = [float(sum(mat[j][i] for j in range(N))) for i in range(N, 2*N)]
        sizes = exp_sizes + imp_sizes  # [US_e, CN_e, ..., US_i, CN_i, ...]

        # Also add interleaved sizes for D3 display: [US_e, US_i, CN_e, CN_i, ...]
        interleaved_sizes = []
        for i in range(N):
            interleaved_sizes.append(exp_sizes[i])
            interleaved_sizes.append(imp_sizes[i])

        result[int(yr)] = {
            "matrix": mat,              # non-interleaved, no reverses
            "arc_sizes": sizes,         # grouped: exports then imports
            "interleaved_sizes": interleaved_sizes,  # for D3 display
        }
    return result

trade_years = sorted(trade_raw["year"].unique())
cite_years = sorted(cite_raw["year"].unique())
cite_overlap = [y for y in cite_years if 1995 <= y <= 2017]
all_years = sorted(set(trade_years) | set(cite_overlap))

trade_data = build_matrices(trade_raw, "trade_value", trade_years, top_individuals, all_countries)
cite_data = build_matrices(cite_raw, "weight", cite_overlap, top_individuals, all_countries)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

output = {
    "countries": all_countries,
    "country_names": {c: iso_to_name.get(c,c) if c!="ROW" else "Rest of World" for c in all_countries},
    "colors": node_colors,
    "trade_years": [int(y) for y in trade_years],
    "citation_years": [int(y) for y in cite_overlap],
    "all_years": [int(y) for y in all_years],
    "trade": {},
    "citation": {},
}

for yr in trade_years:
    yr_key = str(int(yr))
    d = trade_data[int(yr)]
    output["trade"][yr_key] = {
        "matrix": d["matrix"],
        "arc_sizes": [float(v) for v in d["arc_sizes"]],
        "interleaved_sizes": [float(v) for v in d["interleaved_sizes"]],
    }

for yr in cite_overlap:
    yr_key = str(int(yr))
    d = cite_data[int(yr)]
    output["citation"][yr_key] = {
        "matrix": d["matrix"],
        "arc_sizes": [float(v) for v in d["arc_sizes"]],
        "interleaved_sizes": [float(v) for v in d["interleaved_sizes"]],
    }

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(output, f, cls=NpEncoder, indent=None)
print(f"Saved -> {args.output} ({os.path.getsize(args.output)/1024:.0f} KB)")
