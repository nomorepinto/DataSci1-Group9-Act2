"""
Combine multiple semicolon-delimited "wide" export CSVs (one column pair per
year: "<YYYY> Quantity" / "<YYYY> FOB") into a single tidy long-format
DataFrame:

    Commodity | Country Of Destination | Year | Quantity | FOB

This long format is what you want for anomaly detection: one row per
(commodity, country, year), so you can engineer trend features (YoY % change,
z-scores, rolling stats) before feeding it to IsolationForest.

Usage:
    python combine_exports.py file1.csv file2.csv ... -o combined.csv

Or import combine_files() directly in a notebook.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ID_COLS = ["Commodity", "Country Of Destination"]
YEAR_COL_PATTERN = re.compile(r"^(\d{4})\s+(Quantity|FOB)$", re.IGNORECASE)


def load_wide_csv(path: str) -> pd.DataFrame:
    """Read one semicolon-delimited wide CSV."""
    # utf-8-sig strips a BOM if present (common in Excel-exported CSVs on
    # Windows), which otherwise silently corrupts the first column's name.
    df = pd.read_csv(path, sep=";", quotechar='"', dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"').strip() for c in df.columns]
    return df


def wide_to_long(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Melt a single wide-format file into long format."""
    missing_id = [c for c in ID_COLS if c not in df.columns]
    if missing_id:
        raise ValueError(
            f"{source_file}: missing expected columns {missing_id}\n"
            f"  Columns found in file: {list(df.columns)}"
        )

    year_cols = [c for c in df.columns if YEAR_COL_PATTERN.match(c)]
    if not year_cols:
        raise ValueError(f"{source_file}: no '<YYYY> Quantity/FOB' columns found")

    # Melt to long: one row per (id cols, year_col, value)
    long_df = df.melt(
        id_vars=ID_COLS,
        value_vars=year_cols,
        var_name="year_metric",
        value_name="value",
    )

    # Split "2020 Quantity" -> year=2020, metric=Quantity
    extracted = long_df["year_metric"].str.extract(YEAR_COL_PATTERN)
    long_df["Year"] = extracted[0].astype(int)
    long_df["metric"] = extracted[1].str.title()  # Quantity / Fob -> Quantity/Fob
    long_df["metric"] = long_df["metric"].replace({"Fob": "FOB"})

    # Clean numeric values: strip commas/spaces, coerce to float
    long_df["value"] = (
        long_df["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    # Pivot metric (Quantity/FOB) back out into columns
    wide_again = long_df.pivot_table(
        index=ID_COLS + ["Year"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_again.columns.name = None

    wide_again["source_file"] = Path(source_file).name
    return wide_again


def combine_files(paths: list[str]) -> pd.DataFrame:
    """Load and combine multiple wide CSVs into one long DataFrame."""
    frames = []
    for path in paths:
        df = load_wide_csv(path)
        long_df = wide_to_long(df, path)
        frames.append(long_df)
        print(f"  loaded {path}: {len(long_df)} rows, "
              f"years {sorted(long_df['Year'].unique())}")

    combined = pd.concat(frames, ignore_index=True)

    # Files can overlap on (Commodity, Country, Year) if year ranges overlap.
    # Keep the first occurrence but warn so you can decide how to reconcile.
    dupe_mask = combined.duplicated(subset=ID_COLS + ["Year"], keep=False)
    n_dupes = dupe_mask.sum()
    if n_dupes:
        print(f"\nWarning: {n_dupes} rows share the same Commodity/Country/Year "
              f"across files (overlapping year ranges). Keeping first occurrence "
              f"per group; inspect 'source_file' if you need to reconcile manually.")
        combined = combined.sort_values("source_file").drop_duplicates(
            subset=ID_COLS + ["Year"], keep="first"
        )

    combined = combined.sort_values(ID_COLS + ["Year"]).reset_index(drop=True)

    # Ensure consistent column order
    front = ID_COLS + ["Year", "Quantity", "FOB"]
    other = [c for c in combined.columns if c not in front]
    combined = combined[front + other]

    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Wide-format CSV files to combine")
    parser.add_argument("-o", "--output", default="combined_exports.csv",
                         help="Output CSV path (default: combined_exports.csv)")
    args = parser.parse_args()

    print("Loading files...")
    combined = combine_files(args.files)

    combined.to_csv(args.output, index=False)
    print(f"\nDone. Combined dataset: {len(combined)} rows, "
          f"{combined['Year'].nunique()} years "
          f"({combined['Year'].min()}-{combined['Year'].max()}), "
          f"{combined['Commodity'].nunique()} commodities, "
          f"{combined['Country Of Destination'].nunique()} countries.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
