#!/usr/bin/env python3
"""
prepare_token_dataset.py

Read your augmented CSV (with descriptions starting “Mensafood …”),
strip that prefix, and write out a new CSV for textual‑inversion training.
"""

import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv",  required=True, help="meals_augmented.csv")
    p.add_argument("--output_csv", required=True, help="where to save cleaned CSV")
    p.add_argument("--prefix",     default="Mensafood ", help="token prefix to strip")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    # Assume column 'description' holds the prompt
    def strip_prefix(text: str) -> str:
        if isinstance(text, str) and text.startswith(args.prefix):
            return text[len(args.prefix):]
        return text

    df["description"] = df["description"].apply(strip_prefix)
    df.to_csv(args.output_csv, index=False)
    print(f"✓ Wrote cleaned CSV with {len(df)} rows → {args.output_csv}")

if __name__ == "__main__":
    main()
