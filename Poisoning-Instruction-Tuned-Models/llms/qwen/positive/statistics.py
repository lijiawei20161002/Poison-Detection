#!/usr/bin/env python3
import pandas as pd
import argparse
import json
import sys

def compute_and_print(df, col, to_json):
    # Basic descriptive stats
    count = int(df[col].count())
    mean = float(df[col].mean())
    median = float(df[col].median())
    std  = float(df[col].std())
    minimum = float(df[col].min())
    maximum = float(df[col].max())
    q25 = float(df[col].quantile(0.25))
    q75 = float(df[col].quantile(0.75))

    stats = {
        "count": count,
        "mean": mean,
        "median": median,
        "std": std,
        "min": minimum,
        "25%": q25,
        "75%": q75,
        "max": maximum,
    }

    # Print to console
    print(f"Statistics for column '{col}':")
    print(f"  Count   : {count}")
    print(f"  Mean    : {mean:.4f}")
    print(f"  Median  : {median:.4f}")
    print(f"  Std Dev : {std:.4f}")
    print(f"  Min     : {minimum:.4f}")
    print(f"  25%     : {q25:.4f}")
    print(f"  75%     : {q75:.4f}")
    print(f"  Max     : {maximum:.4f}")

    # Optionally write out to JSON
    if to_json:
        with open(to_json, "w", encoding="utf-8") as fout:
            json.dump(stats, fout, indent=2)
        print(f"\n✅ Stats written to {to_json}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for a numeric column in a CSV."
    )
    parser.add_argument(
        "--csv", type=str, default="influence_average_scores.csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--col", type=str, default="average_score",
        help="Name of the numeric column to summarize."
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="If set, write the stats to this JSON file."
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"❌ Failed to read CSV '{args.csv}': {e}", file=sys.stderr)
        sys.exit(1)

    if args.col not in df.columns:
        print(f"❌ Column '{args.col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(1)

    compute_and_print(df, args.col, args.json)

if __name__ == "__main__":
    main()