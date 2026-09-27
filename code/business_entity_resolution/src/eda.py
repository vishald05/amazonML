"""Quick exploratory data analysis script."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from io_utils import load_source_data, load_ground_truth

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading data...")
train_s1 = load_source_data(TRAIN_SOURCE1, "train_source1")
train_s2 = load_source_data(TRAIN_SOURCE2, "train_source2")
train_s3 = load_source_data(TRAIN_SOURCE3, "train_source3")
ground_truth = load_ground_truth(TRAIN_GROUND_TRUTH)

print("\n" + "=" * 80)
print("DATASET SIZES")
print("=" * 80)
print(f"Train Source 1: {len(train_s1):,} entities")
print(f"Train Source 2: {len(train_s2):,} entities")
print(f"Train Source 3: {len(train_s3):,} entities")
print(f"Ground truth:   {len(ground_truth):,} S1 entities")

# Ground truth analysis
total_matches = sum(len(v) for v in ground_truth.values())
singletons = sum(1 for v in ground_truth.values() if len(v) == 0)
multi_match = sum(1 for v in ground_truth.values() if len(v) > 1)

print("\n" + "=" * 80)
print("GROUND TRUTH STATISTICS")
print("=" * 80)
print(f"Total matches: {total_matches:,}")
print(f"Avg matches per S1: {total_matches/len(ground_truth):.2f}")
print(f"Singletons: {singletons:,} ({100*singletons/len(ground_truth):.1f}%)")
print(f"Multi-match entities: {multi_match:,} ({100*multi_match/len(ground_truth):.1f}%)")

# Distribution of matches
match_counts = [len(v) for v in ground_truth.values()]
print(f"\nMatch count distribution:")
print(f"  0 matches: {sum(1 for c in match_counts if c == 0):,}")
print(f"  1 match:   {sum(1 for c in match_counts if c == 1):,}")
print(f"  2 matches: {sum(1 for c in match_counts if c == 2):,}")
print(f"  3 matches: {sum(1 for c in match_counts if c == 3):,}")
print(f"  4+ matches: {sum(1 for c in match_counts if c >= 4):,}")
print(f"  Max matches: {max(match_counts)}")

# Country distribution
print("\n" + "=" * 80)
print("COUNTRY DISTRIBUTION")
print("=" * 80)
print("\nSource 1:")
print(train_s1['country'].value_counts())
print("\nSource 2:")
print(train_s2['country'].value_counts())
print("\nSource 3:")
print(train_s3['country'].value_counts())

# Missing values
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

def check_missing(df, name):
    print(f"\n{name}:")
    for col in ['business_name', 'business_address', 'country']:
        missing = df[col].isna().sum()
        empty = (df[col] == '').sum()
        print(f"  {col}: {missing:,} NaN, {empty:,} empty")

check_missing(train_s1, "Source 1")
check_missing(train_s2, "Source 2")
check_missing(train_s3, "Source 3")

# Sample records
print("\n" + "=" * 80)
print("SAMPLE RECORDS")
print("=" * 80)
print("\nSource 1 samples:")
print(train_s1.head(3).to_string())

print("\nSource 2 samples:")
print(train_s2.head(3).to_string())

print("\nSource 3 samples:")
print(train_s3.head(3).to_string())

# Match examples
print("\n" + "=" * 80)
print("SAMPLE MATCHES")
print("=" * 80)

# Find an entity with multiple matches
for s1_id, matches in list(ground_truth.items())[:20]:
    if len(matches) >= 3:
        print(f"\nS1 Entity: {s1_id}")
        s1_rec = train_s1[train_s1['entity_id'] == s1_id].iloc[0]
        print(f"  Name: {s1_rec['business_name']}")
        print(f"  Address: {s1_rec['business_address']}")
        print(f"  Country: {s1_rec['country']}")
        print(f"  Matches ({len(matches)}):")

        for match_id in list(matches)[:3]:
            if match_id.startswith('S2-'):
                match_rec = train_s2[train_s2['entity_id'] == match_id]
            else:
                match_rec = train_s3[train_s3['entity_id'] == match_id]

            if len(match_rec) > 0:
                match_rec = match_rec.iloc[0]
                print(f"    {match_id}:")
                print(f"      Name: {match_rec['business_name']}")
                print(f"      Address: {match_rec['business_address']}")
                print(f"      Country: {match_rec['country']}")
        break

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
