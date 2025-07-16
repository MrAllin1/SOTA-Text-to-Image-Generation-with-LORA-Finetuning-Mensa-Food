
import sys
from pathlib import Path
import pandas as pd

input_csv="/Users/andialidema/Desktop/Freiburg UNI/semester 2/dl lab/project/mensa_t2i/data/meals_unique.csv"
output_csv="/Users/andialidema/Desktop/Freiburg UNI/semester 2/dl lab/project/mensa_t2i/data/meals_unique_mensafood.csv"

def prepend_token(src_path: Path, dst_path: Path) -> None:
    """Read CSV, add 'mensafood ' to description, save new CSV."""
    df = pd.read_csv(src_path)
    df["description"] = "Mensafood " + df["description"].astype(str)
    df.to_csv(dst_path, index=False)
    print(f"✓ Prepended token to {len(df)} rows → {dst_path}")

def main() -> None:
    src = Path(input_csv)
    if not src.exists():
        sys.exit(f"❌ Input file not found: {src}")

    # Auto‑name output if left blank
    dst = Path(output_csv) if output_csv else src.with_stem(src.stem + "_mensafood")

    prepend_token(src, dst)

if __name__ == "__main__":
    main()