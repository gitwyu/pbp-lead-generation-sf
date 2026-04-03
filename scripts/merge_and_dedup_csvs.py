import argparse
import os
import sys

import numpy as np
import pandas as pd


def main():
  parser = argparse.ArgumentParser(
      description="Merge and deduplicate two CSVs by Company Name."
  )
  parser.add_argument("csv_a", help="Path to the first CSV file (Option A)")
  parser.add_argument("csv_b", help="Path to the second CSV file (Option B)")
  parser.add_argument(
      "--output",
      default="merged_leads.csv",
      help="Path to save the output CSV file.",
  )
  args = parser.parse_args()

  if not os.path.exists(args.csv_a):
    print(f"Error: {args.csv_a} does not exist.")
    sys.exit(1)
  if not os.path.exists(args.csv_b):
    print(f"Error: {args.csv_b} does not exist.")
    sys.exit(1)

  print(f"Reading {args.csv_a}...")
  df_a = pd.read_csv(args.csv_a)
  print(f"Reading {args.csv_b}...")
  df_b = pd.read_csv(args.csv_b)

  if "Company Name" not in df_a.columns or "Company Name" not in df_b.columns:
    print("Error: Both CSVs must contain a 'Company Name' column.")
    sys.exit(1)

  # Basic deduplication within the same file just in case
  df_a = df_a.drop_duplicates(subset=["Company Name"], keep="first")
  df_b = df_b.drop_duplicates(subset=["Company Name"], keep="first")

  # Standardize all columns to preserve order
  all_cols = list(df_a.columns) + [
      c for c in df_b.columns if c not in df_a.columns
  ]

  # Clean the company name column for easier matching
  df_a["_match_name"] = df_a["Company Name"].astype(str).str.strip().str.lower()
  df_b["_match_name"] = df_b["Company Name"].astype(str).str.strip().str.lower()

  names_a = set(df_a["_match_name"].dropna())
  names_b = set(df_b["_match_name"].dropna())

  common_names = names_a.intersection(names_b)
  unique_a = names_a - common_names
  unique_b = names_b - common_names

  final_rows = []

  # Add unique rows directly
  final_rows.extend(df_a[df_a["_match_name"].isin(unique_a)].to_dict("records"))
  final_rows.extend(df_b[df_b["_match_name"].isin(unique_b)].to_dict("records"))

  print(f"\nFound {len(unique_a)} unique to A, {len(unique_b)} unique to B.")
  print(f"Found {len(common_names)} overlapping companies to check...")

  # Resolve common rows
  for name in common_names:
    row_a = df_a[df_a["_match_name"] == name].iloc[0].to_dict()
    row_b = df_b[df_b["_match_name"] == name].iloc[0].to_dict()

    conflicts = []
    merged_row = {}

    for col in all_cols:
      val_a = row_a.get(col, np.nan)
      val_b = row_b.get(col, np.nan)

      str_a = (
          str(val_a).strip()
          if pd.notna(val_a) and str(val_a).strip() not in ("", "nan", "None")
          else None
      )
      str_b = (
          str(val_b).strip()
          if pd.notna(val_b) and str(val_b).strip() not in ("", "nan", "None")
          else None
      )

      # Check for actual conflicts
      if str_a and str_b and str_a != str_b:
        conflicts.append((col, str_a, str_b))
        merged_row[col] = str_a  # temporary assignment
      elif str_a:
        merged_row[col] = str_a
      elif str_b:
        merged_row[col] = str_b
      else:
        merged_row[col] = ""

    # If there are conflicting fields, check for General Contact Webpage first
    if conflicts:
      val_wp_a = row_a.get("General Contact Webpage", np.nan)
      val_wp_b = row_b.get("General Contact Webpage", np.nan)

      str_wp_a = str(val_wp_a).strip() if pd.notna(val_wp_a) and str(val_wp_a).strip() not in ("", "nan", "None") else None
      str_wp_b = str(val_wp_b).strip() if pd.notna(val_wp_b) and str(val_wp_b).strip() not in ("", "nan", "None") else None

      if str_wp_a and not str_wp_b:
        print(f"\n--- Auto-resolving Conflict for Company: '{row_a['Company Name']}' ---")
        print(f"  Choosing [A] because it has General Contact Webpage.")
        final_rows.append(row_a)
      elif str_wp_b and not str_wp_a:
        print(f"\n--- Auto-resolving Conflict for Company: '{row_a['Company Name']}' ---")
        print(f"  Choosing [B] because it has General Contact Webpage.")
        final_rows.append(row_b)
      else:
        print(f"\n--- Conflict for Company: '{row_a['Company Name']}' ---")
        for col, va, vb in conflicts:
          print(f"  Field: {col}")
          print(f"    [A] ({args.csv_a}): {va}")
          print(f"    [B] ({args.csv_b}): {vb}")

        choice = ""
        while choice not in ["a", "b"]:
          choice = input("Choose version to keep [A/B]: ").strip().lower()

        if choice == "a":
          final_rows.append(row_a)
        else:
          final_rows.append(row_b)
    else:
      # No conflicts, safely append the merged row
      final_rows.append(merged_row)

  final_df = pd.DataFrame(final_rows)

  if "_match_name" in final_df.columns:
    final_df = final_df.drop(columns=["_match_name"])

  # Ensure original column order
  ordered_cols = [c for c in all_cols if c in final_df.columns]
  final_df = final_df[ordered_cols]

  final_df.to_csv(args.output, index=False)
  print(
      f"\nSuccessfully merged {len(df_a)} and {len(df_b)} rows into"
      f" {len(final_df)} rows."
  )
  print(f"Saved clean dataset to {args.output}")


if __name__ == "__main__":
  main()
