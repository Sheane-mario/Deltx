import pandas as pd
from pathlib import Path

input_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "train_balanced.parquet"
output_path = input_path.with_suffix(".csv")

df = pd.read_parquet(input_path)
df.to_csv(output_path, index=False)

print(f"Converted {input_path.name} -> {output_path.name}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")
