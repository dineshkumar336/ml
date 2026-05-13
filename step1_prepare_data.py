"""
STEP 1 — prepare_data.py
========================
Converts your existing labeled_complaints.csv → clean train/val/test splits
ready for the new model.

Your existing train_dataset.csv and test_dataset.csv are also compatible —
this script just re-validates them and optionally re-splits from scratch.

Run:
    python step1_prepare_data.py

Output files:
    data/train_new.csv   (~70% of data)
    data/val_new.csv     (~15% of data)
    data/test_new.csv    (~15% of data)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)

# ── Load your original labeled dataset ───────────────────────────────────────
print("Loading labeled_complaints.csv ...")
df = pd.read_csv("labeled_complaints.csv")

print(f"Total rows        : {len(df)}")
print(f"Columns           : {df.columns.tolist()}")
print(f"Priority dist     :\n{df['Priority_Level'].value_counts()}")
print(f"Urgency range     : {df['Urgency_Score'].min()} – {df['Urgency_Score'].max()}")
print(f"Sectors           : {df['Sector'].unique().tolist()}")
print()

# ── Validate & clean ──────────────────────────────────────────────────────────
REQUIRED_COLS = ["Complaint_Narrative", "Product", "Sector",
                 "Priority_Level", "Urgency_Score"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in your CSV: {missing}")

df = df.dropna(subset=REQUIRED_COLS)
df["Complaint_Narrative"] = df["Complaint_Narrative"].astype(str).str.strip()
df["Product"]             = df["Product"].astype(str).str.strip()
df["Sector"]              = df["Sector"].astype(str).str.strip()
df["Priority_Level"]      = df["Priority_Level"].str.strip().str.capitalize()
df["Urgency_Score"]       = df["Urgency_Score"].clip(1.0, 10.0)

# Verify priority values
valid_priorities = {"Low", "Medium", "High"}
bad = df[~df["Priority_Level"].isin(valid_priorities)]
if len(bad) > 0:
    print(f"WARNING: {len(bad)} rows have unexpected Priority_Level values:")
    print(bad["Priority_Level"].value_counts())
    df = df[df["Priority_Level"].isin(valid_priorities)]
    print(f"Rows after filtering: {len(df)}")

print(f"Clean dataset size: {len(df)}")

# ── Split: 70% train | 15% val | 15% test (stratified by Priority_Level) ─────
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["Priority_Level"], random_state=42)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["Priority_Level"], random_state=42)

# Keep only the 5 required columns (drop old combined_text / label if present)
KEEP_COLS = ["Complaint_Narrative", "Product", "Sector",
             "Priority_Level", "Urgency_Score"]
train_df = train_df[KEEP_COLS].reset_index(drop=True)
val_df   = val_df[KEEP_COLS].reset_index(drop=True)
test_df  = test_df[KEEP_COLS].reset_index(drop=True)

train_df.to_csv("data/train_new.csv", index=False)
val_df.to_csv("data/val_new.csv",     index=False)
test_df.to_csv("data/test_new.csv",   index=False)

print()
print("✅ Data splits saved:")
print(f"   data/train_new.csv  → {len(train_df)} rows")
print(f"   data/val_new.csv    → {len(val_df)} rows")
print(f"   data/test_new.csv   → {len(test_df)} rows")
print()
print("Priority distribution in each split:")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    counts = split["Priority_Level"].value_counts()
    print(f"  {name:5s}: Low={counts.get('Low',0)}  Medium={counts.get('Medium',0)}  High={counts.get('High',0)}")

print()
print("➡  Next: run  python step2_train.py")
