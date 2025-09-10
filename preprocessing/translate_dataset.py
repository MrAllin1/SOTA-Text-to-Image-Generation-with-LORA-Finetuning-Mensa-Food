#!/usr/bin/env python3
"""
translate_meals_google.py
-------------------------------------
Batch‑translates a German Mensa menu CSV to English
while keeping the `mensa` and `image_path` columns intact.

USAGE:
    python translate_meals_google.py input.csv
OUTPUT:
    input_en.csv  (same folder)
"""

import sys
import time
import random
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator

# ---------- CONFIG ----------------------------------------------------------
BATCHSIZE = 40             # adjust if you like
PAUSE     = (1.0, 2.0)     # random sleep between batches (helps avoid 429s)

# Fixed‑value label maps
TYPE_MAP = {
    "Essen 1":          "Meal 1",
    "Essen 2":          "Meal 2",
    "Essen 3":          "Meal 3",
    "Abendessen 1":     "Dinner 1",
    "Abendessen 2":     "Dinner 2",
    "Tagesgericht":     "Daily dish",
    "Wochenangebot":    "Weekly special",
    "Schneller Teller": "Quick plate",
}
DIET_MAP = {
    "Vegetarisch":          "Vegetarian",
    "Vegan":                "Vegan",
    "Vegan auf Anfrage":    "Vegan (on request)",
    "Nicht-vegetarisch":    "Non‑vegetarian",
}
# ---------------------------------------------------------------------------


def translate_series(series: pd.Series, translator: GoogleTranslator) -> pd.Series:
    """
    Batch‑translate a pandas Series of German strings to English.
    Cleans NaN/None/ints, retries row‑by‑row on failure.
    """
    out = series.copy()

    for start in range(0, len(series), BATCHSIZE):
        # Slice the batch and coerce EVERYTHING to str ("" for NaN)
        batch_raw = series.iloc[start:start + BATCHSIZE]
        batch     = ["" if pd.isna(x) else str(x) for x in batch_raw]

        try:
            translated = translator.translate_batch(batch)
            out.iloc[start:start + BATCHSIZE] = translated
        except Exception as e:
            print(f"⚠️ Batch {start}-{start+len(batch)-1} failed ({e}); retrying row‑by‑row.")
            for i, txt in enumerate(batch, start=start):
                try:
                    out.iat[i] = translator.translate(txt)
                except Exception as e2:
                    print(f"  ⚠️ Row {i} still failed ({e2}) – leaving original text.")
                    out.iat[i] = txt
            # gentle back‑off before next batch
            time.sleep(random.uniform(4, 7))
            continue

        time.sleep(random.uniform(*PAUSE))

    return out


def main():
    if len(sys.argv) != 2:
        print("Usage: python translate_meals_google.py <input_csv>")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists() or src.suffix.lower() != ".csv":
        print("❌  Please supply a valid CSV file path.")
        sys.exit(1)

    # ---------- LOAD ----------
    df = pd.read_csv(src)

    # Ensure required column exists
    if "description" not in df.columns:
        print("❌  Column 'description' not found in CSV.")
        sys.exit(1)

    # Coerce description to clean strings early
    df["description"] = df["description"].fillna("").astype(str)

    # ---------- TRANSLATE ----------
    translator = GoogleTranslator(source="de", target="en")

    df["description"] = translate_series(df["description"], translator)
    df["type"] = df["type"].replace(TYPE_MAP)
    df["diet"] = df["diet"].replace(DIET_MAP)

    # ---------- SAVE ----------
    dst = src.with_name(src.stem + "_en.csv")
    df.to_csv(dst, index=False)
    print(f"✅  Saved: {dst}")

if __name__ == "__main__":
    main()
