#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(csv_path: str, output_png: str):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Create a histogram (bar‐chart style) of the average_score distribution
    plt.figure()
    plt.hist(df['average_score'], bins=20)
    plt.xlabel('Average Score')
    plt.ylabel('Count')
    plt.title('Distribution of Average Scores')

    # Save as PNG
    plt.savefig(output_png)
    plt.close()
    print(f"✅ Bar chart saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the distribution of 'average_score' from a CSV and save as PNG."
    )
    parser.add_argument(
        "--csv", type=str, default="influence_average_scores.csv",
        help="Path to the input CSV (must contain an 'average_score' column)."
    )
    parser.add_argument(
        "--out", type=str, default="distribution.png",
        help="Filename for the output PNG."
    )
    args = parser.parse_args()
    main(args.csv, args.out)